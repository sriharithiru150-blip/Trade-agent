"""AI trading agent — Claude-powered orchestrator (event-driven architecture).

Signal hierarchy (highest edge → lowest edge)
----------------------------------------------
1. Exchange-filed corporate events  (board outcomes, results, M&A, buybacks)
   — milliseconds after filing, before most retail traders read a headline.
2. Institutional bulk/block deals   (FII, DII, promoter conviction flow)
   — same-day NSE filings showing where big money is moving.
3. Options positioning              (PCR, max pain, unusual OI buildup)
   — forward-looking; institutions hedge/position before moves.
4. Technical context                (trend, RSI, volume)
   — secondary confirmation only; never the primary thesis.

What was removed vs. the previous version
------------------------------------------
- NewsAPI / Google RSS news (stale by the time we read it)
- Pure RSI/MACD signal blending as the primary alpha source
- 30-minute blind polling regardless of market activity

What replaced it
-----------------
- NSE corporate announcement polling (every cycle, new events only)
- NSE bulk/block deal scraping
- NSE options chain for PCR, max pain, unusual OI
- Liquidity filter + slippage model on every position before entry
- Swing/positional holding (2-10 sessions) instead of forced intraday
- Claude reasons over structured evidence, not free-form sentiment
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from trade_agent.analysis.event_scorer import EventScorer
from trade_agent.analysis.liquidity import compute_liquidity_profile, compute_slippage
from trade_agent.analysis.technical import compute_technical_snapshot
from trade_agent.config import Settings
from trade_agent.data.company_graph import CompanyGraph
from trade_agent.data.events import EventScraper
from trade_agent.data.market_data import MarketDataClient
from trade_agent.data.options_chain import OptionsChainClient
from trade_agent.execution.order_executor import KillSwitchError, OrderExecutor
from trade_agent.execution.portfolio_manager import PortfolioManager
from trade_agent.strategies.event_driven import EventDrivenStrategy
from trade_agent.utils.logger import get_logger
from trade_agent.utils.market_hours import is_market_open, now_ist

log = get_logger(__name__)

# ── Tool definitions for Claude ──────────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_market_overview",
        "description": (
            "Get a snapshot of the broader Indian market: Nifty 50, Sensex, "
            "Bank Nifty, and India VIX (fear index). Returns price, day change %, "
            "and a market-breadth assessment. Always call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_corporate_events",
        "description": (
            "Fetch recent NSE corporate announcements for a symbol or for the "
            "entire market (symbol=null). Returns only NEW events filed since "
            "the last poll — no stale data. Events include: earnings results, "
            "board meeting outcomes, buybacks, M&A, QIPs, SEBI orders, "
            "management changes, promoter pledging. "
            "This is the PRIMARY alpha signal — call before anything else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol to filter (e.g. RELIANCE), or omit for all symbols",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_institutional_flows",
        "description": (
            "Fetch today's NSE bulk deals and block deals. "
            "Shows which institutional clients (FII, DII, mutual funds, promoters) "
            "are buying or selling, and the INR value. "
            "FII/promoter buying is a strong conviction signal. "
            "Institutional selling at scale is a strong warning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Filter to a specific NSE symbol, or omit for all",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_options_positioning",
        "description": (
            "Fetch the NSE options chain for a symbol and compute: "
            "Put-Call Ratio (PCR), max pain strike, call/put OI walls (key "
            "support/resistance levels), IV skew (put premium vs call premium), "
            "and any unusual OI buildup at specific strikes. "
            "PCR > 1.2 is contrarian bullish. PCR < 0.7 is contrarian bearish. "
            "Unusual OI may indicate informed positioning ahead of an event."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "NSE equity symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_stock_technicals",
        "description": (
            "Fetch the current quote and compute technical indicators "
            "(RSI, MACD, Bollinger Bands, ATR, volume ratio, trend direction). "
            "Use as SECONDARY confirmation only — not as a standalone trade thesis. "
            "Also returns the liquidity profile: ADV, bid-ask spread estimate, "
            "impact cost, and maximum safe position size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "NSE symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "score_opportunity",
        "description": (
            "Run the EventScorer to compute a structured conviction score (0–1) "
            "that combines event urgency, institutional flow direction, and options "
            "positioning for a symbol. Call this after gathering events, flows, and "
            "options data for a symbol to get a quantified summary before deciding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "NSE symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio_status",
        "description": (
            "Return the current portfolio: open positions, unrealised P&L, "
            "realised P&L, and available capacity. Check this before executing "
            "any new trade to verify position limits."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "execute_trade",
        "description": (
            "Open a LONG position. A stop-loss order is placed immediately after "
            "entry. Only call this when ALL of the following are true:\n"
            "- A material corporate event is the primary thesis\n"
            "- Conviction score ≥ 0.60\n"
            "- Stock passes liquidity filter\n"
            "- Risk:reward is at least 2:1 AFTER accounting for break-even cost\n"
            "- Portfolio has capacity\n"
            "In DRY_RUN mode this is a paper trade — no real money at risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "quantity": {"type": "integer"},
                "entry_price": {"type": "number"},
                "stop_loss": {"type": "number"},
                "take_profit": {"type": "number"},
                "holding_period_days": {
                    "type": "integer",
                    "description": "Expected holding period in trading sessions (1-10)",
                },
                "thesis": {
                    "type": "string",
                    "description": "One-paragraph explanation of the trade thesis",
                },
            },
            "required": [
                "symbol", "quantity", "entry_price",
                "stop_loss", "take_profit", "thesis",
            ],
        },
    },
    {
        "name": "close_position",
        "description": "Close an existing open position at the current market price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["symbol", "reason"],
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_SWING = """You are an expert AI trading agent specialising in Indian equity markets (NSE/BSE).
Operating in SWING mode: target 2-10 session holds, ATR-based stops (1.5×ATR14 daily), freshness 15/45/90 min.

## Your edge
You trade on INFORMATION EVENTS filed directly with stock exchanges — not on lagging technical patterns or
stale news. Your alpha comes from being faster and more thorough than retail traders at processing
exchange-filed material events, cross-referencing them with institutional flow data and options positioning.

## Analysis process (follow this order)
1. `get_market_overview` — assess Nifty/Sensex/VIX. If Nifty is down >1.5% or VIX > 22, raise
   the conviction bar to ≥ 0.72 before entering any trade.
2. `get_corporate_events` (no symbol filter) — scan for any new material events across the market.
3. For each symbol with a material event:
   a. `get_institutional_flows` — is smart money moving in the same direction?
   b. `get_options_positioning` — does options market corroborate or contradict?
   c. `get_stock_technicals` — check liquidity, trend, and compute break-even cost.
   d. `score_opportunity` — get the quantified conviction score.
4. `get_portfolio_status` — check capacity before entering.
5. Only enter if: event is material + conviction ≥ 0.60 + liquidity passes + R:R ≥ 2:1 net of costs.

## What NOT to do
- Do NOT trade purely on RSI/MACD signals with no event catalyst.
- Do NOT trade on news from public feeds — it is already priced in.
- Do NOT enter stocks with ADV < ₹20 Cr — slippage will destroy the edge.
- Do NOT open more than the configured max_open_positions simultaneously.
- Do NOT ignore the liquidity-adjusted break-even cost when setting stop-loss.

## Risk rules (non-negotiable)
- Stop-loss MUST be set below the break-even cost level (stop-loss < entry × (1 - break_even_pct)).
- Minimum risk:reward = 2:1 AFTER transaction costs.
- If India VIX > 22: reduce position size by 50%.
- If Nifty is down > 2% intraday: no new entries regardless of individual signals.
- For swing holds: apply configured take-profit/stop-loss targets; EOD square-off by the configured time.

## Output format (after completing analysis)
Provide a structured summary:
1. Market conditions and breadth
2. Events detected and their classification
3. Symbols analysed with conviction scores
4. Trades taken (or explicitly skipped with reason)
5. Current portfolio status
6. Any elevated risks or observations
"""

_SYSTEM_PROMPT_INTRADAY = """You are an expert AI trading agent specialising in Indian equity markets (NSE/BSE).
Operating in INTRADAY mode: all positions MUST be closed before EOD square-off trigger.
ATR computed from 5-min candles. Stop: ~0.5% (ATR-based). Target: ~1.2% (2.4:1 R:R min).
Freshness windows: Nifty50=5 min, mid-cap=15 min, small-cap=30 min.
Options data stale after 2 min in intraday mode. ADV floor: ₹50 Cr.

## Your edge
You react to SAME-DAY exchange events that move intraday price — pre-market bulk deals filed before 9:15,
intraday block deal window (9:15–9:50), and unusual intraday options OI buildup ahead of a catalyst.
Your advantage is processing these signals faster than retail traders.

## Analysis process (follow this order)
1. `get_market_overview` — assess Nifty/Sensex/VIX. If Nifty is down >1.5% or VIX > 22, raise
   the conviction bar to ≥ 0.75 before entering. No new entries if Nifty down > 2%.
2. `get_corporate_events` (no symbol filter) — scan for events within the 5/15/30 min intraday window.
   Events older than the intraday window are already priced — skip them.
3. For each symbol with a fresh intraday event:
   a. `get_institutional_flows` — pre-market bulk deals are the strongest intraday signal.
   b. `get_options_positioning` — check PCR and unusual OI buildup. NOTE: options data may
      be up to 2 min stale; treat with caution during fast-moving events.
   c. `get_stock_technicals` — verify ADV ≥ ₹50 Cr. Check 5-min ATR for stop placement.
   d. `score_opportunity` — conviction must be ≥ 0.65 for intraday (tighter than swing).
4. `get_portfolio_status` — check capacity. Intraday: max 3 simultaneous positions recommended.
5. Only enter if: fresh event + conviction ≥ 0.65 + ADV ≥ ₹50 Cr + R:R ≥ 2.4:1 after costs.

## What NOT to do
- Do NOT enter on events older than the intraday freshness window — they are already priced.
- Do NOT enter stocks with ADV < ₹50 Cr intraday — spread will eat the target.
- Do NOT let any position survive past the EOD square-off trigger — no exceptions.
- Do NOT use daily RSI/MACD as intraday signals — use 5-min price action and event flow only.
- Do NOT enter if fewer than 90 minutes remain until market close.

## Risk rules (non-negotiable)
- Stop-loss from 5-min ATR (1.5×ATR). Maximum stop 0.8% for intraday.
- Minimum risk:reward = 2.4:1 AFTER transaction costs (target ≥ 1.2% for 0.5% stop).
- If India VIX > 22: no new intraday entries.
- If Nifty is down > 1.5% intraday: no new entries.
- ALL positions must be closed at or before the EOD square-off time — hard rule.
- If a position hits stop-loss, do NOT re-enter the same stock that session.

## Output format (after completing analysis)
Provide a structured summary:
1. Market conditions (Nifty, VIX, breadth)
2. Fresh intraday events found and their age/tier
3. Symbols analysed with conviction scores and ATR stop levels
4. Trades taken (or explicitly skipped with reason)
5. Current portfolio status and time to EOD cutoff
6. Any elevated risks
"""


def _build_system_prompt(intraday_mode: bool) -> str:
    return _SYSTEM_PROMPT_INTRADAY if intraday_mode else _SYSTEM_PROMPT_SWING


# ── Agent ─────────────────────────────────────────────────────────────────────


class TradingAgent:
    """Claude-orchestrated trading agent with event-driven signal architecture.

    Args:
        settings: Application settings loaded from environment.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # Data clients
        self._market_data = MarketDataClient()
        self._event_scraper = EventScraper(max_age_minutes=120)
        self._options_client = OptionsChainClient()
        self._company_graph = CompanyGraph()

        # Analysis
        self._scorer = EventScorer()

        # Execution
        self._executor = OrderExecutor(
            dry_run=settings.dry_run,
            kill_switch=settings.kill_switch,
            kite_api_key=settings.kite_api_key,
            kite_access_token=settings.kite_access_token,
        )
        self._portfolio = PortfolioManager(
            max_open_positions=settings.max_open_positions,
            max_trade_amount=settings.max_trade_amount,
            max_sector_positions=settings.max_sector_positions,
        )

        # Strategy
        self._strategy = EventDrivenStrategy(
            min_conviction=settings.min_signal_score,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_trade_amount=settings.max_trade_amount,
            intraday_mode=settings.intraday_mode,
        )

        # Per-cycle cache so tools don't re-fetch within the same run
        self._cycle_cache: dict[str, Any] = {}

        log.info(
            "agent_initialised",
            model=settings.claude_model,
            dry_run=settings.dry_run,
            watchlist=settings.get_watchlist(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run_once(self) -> str:
        """Run one observe→decide→act cycle.

        Returns:
            Final text summary produced by Claude.
        """
        if not is_market_open():
            log.info("market_closed", time=now_ist().isoformat())
            return "Market is currently closed. No analysis performed."

        # Clear per-cycle cache
        self._cycle_cache = {}

        log.info("agent_cycle_start", time=now_ist().isoformat())
        watchlist = self._settings.get_watchlist()

        ist_now = now_ist()
        mode = "INTRADAY" if self._settings.intraday_mode else "SWING"
        initial_message = (
            f"Run a full event-driven market analysis cycle.\n"
            f"Mode: {mode}\n"
            f"Watchlist: {', '.join(watchlist)}\n"
            f"Current IST time: {ist_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"DRY_RUN: {self._settings.dry_run}\n"
            f"EOD square-off: {self._settings.eod_square_off_minutes} min before close\n\n"
            f"Start with get_market_overview, then scan for new corporate events "
            f"across the full market. Investigate any material events on watchlist "
            f"symbols or any symbol with significant institutional flow."
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
        final_text = ""
        system_prompt = _build_system_prompt(self._settings.intraday_mode)

        for _ in range(25):  # hard cap on iterations
            response = self._client.messages.create(
                model=self._settings.claude_model,
                max_tokens=4096,
                system=system_prompt,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        log.info("agent_cycle_complete")
        return final_text

    def square_off_all(self) -> dict[str, Any]:
        """Close all open positions (end-of-day or emergency).

        Returns:
            Summary dict with closed trade count and total P&L.
        """
        prices: dict[str, float] = {}
        for sym in self._portfolio.open_positions:
            try:
                prices[sym] = self._market_data.get_quote(sym).price
            except Exception:
                prices[sym] = self._portfolio.open_positions[sym].entry_price

        for sym, pos in self._portfolio.open_positions.items():
            self._executor.place_market_order(
                sym, "SELL", pos.quantity, prices.get(sym, pos.entry_price),
                notes="EOD/emergency square-off",
            )

        trades = self._portfolio.close_all_positions(prices, reason="eod")
        total_pnl = sum(t.pnl for t in trades)
        log.info("square_off_complete", trades=len(trades), total_pnl=round(total_pnl, 2))
        return {
            "trades_closed": len(trades),
            "total_pnl": round(total_pnl, 2),
            "trades": [t.to_dict() for t in trades],
        }

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            match name:
                case "get_market_overview":
                    return self._tool_market_overview()
                case "get_corporate_events":
                    return self._tool_corporate_events(inputs.get("symbol"))
                case "get_institutional_flows":
                    return self._tool_institutional_flows(inputs.get("symbol"))
                case "get_options_positioning":
                    return self._tool_options_positioning(inputs["symbol"])
                case "get_stock_technicals":
                    return self._tool_stock_technicals(inputs["symbol"])
                case "score_opportunity":
                    return self._tool_score_opportunity(inputs["symbol"])
                case "get_portfolio_status":
                    return self._tool_portfolio_status()
                case "execute_trade":
                    return self._tool_execute_trade(inputs)
                case "close_position":
                    return self._tool_close_position(inputs["symbol"], inputs.get("reason", "agent"))
                case _:
                    return {"error": f"Unknown tool: {name}"}
        except KillSwitchError as exc:
            return {"error": f"KILL SWITCH ACTIVE: {exc}"}
        except Exception as exc:
            log.error("tool_error", tool=name, error=str(exc))
            return {"error": str(exc)}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_market_overview(self) -> dict[str, Any]:
        from trade_agent.data.market_data import INDEX_TICKERS
        import yfinance as yf

        overview: dict[str, Any] = {}
        for name, ticker in INDEX_TICKERS.items():
            try:
                info = yf.Ticker(ticker).fast_info
                price = float(info.last_price)
                prev = float(info.previous_close)
                chg = round(((price - prev) / prev) * 100, 3) if prev else 0.0
                overview[name] = {
                    "price": price,
                    "change_pct": chg,
                    "day_high": float(info.day_high),
                    "day_low": float(info.day_low),
                }
            except Exception as exc:
                overview[name] = {"error": str(exc)}

        # Market assessment
        nifty = overview.get("NIFTY50", {})
        nifty_chg = nifty.get("change_pct", 0.0)
        vix = overview.get("INDIA_VIX", {}).get("price", 15.0)
        assessment = "normal"
        if nifty_chg < -2.0 or vix > 25:
            assessment = "risk_off_high_caution"
        elif nifty_chg < -1.0 or vix > 20:
            assessment = "risk_off_moderate"
        elif nifty_chg > 1.0 and vix < 15:
            assessment = "risk_on"

        return {
            "indices": overview,
            "assessment": assessment,
            "nifty_change_pct": nifty_chg,
            "india_vix": vix,
            "note": (
                "If assessment is risk_off_high_caution, raise conviction bar to ≥0.72 "
                "and reduce position sizes by 50%. No new entries if Nifty < -2%."
            ),
        }

    def _tool_corporate_events(self, symbol: str | None) -> dict[str, Any]:
        cache_key = f"events_{symbol or 'all'}"
        if cache_key in self._cycle_cache:
            return self._cycle_cache[cache_key]

        events = self._event_scraper.get_announcements(
            symbol=symbol, new_only=True
        )
        mode = "intraday" if self._settings.intraday_mode else "swing"
        result = {
            "count": len(events),
            "events": [e.to_dict() for e in events],
            "mode": mode,
            "note": (
                "Events filed directly with NSE. "
                f"Freshness windows ({mode} mode): "
                "Nifty50=5m/mid=15m/small=30m for intraday, "
                "Nifty50=15m/mid=45m/small=90m for swing. "
                "is_fresh_intraday and is_fresh fields in each event show whether "
                "it is within the relevant window."
            ),
        }
        self._cycle_cache[cache_key] = result
        return result

    def _tool_institutional_flows(self, symbol: str | None) -> dict[str, Any]:
        cache_key = f"flows_{symbol or 'all'}"
        if cache_key in self._cycle_cache:
            return self._cycle_cache[cache_key]

        if symbol:
            deals = self._event_scraper.get_deals_for_symbol(symbol)
        else:
            deals = self._event_scraper.get_all_deals()

        inst_deals = [d for d in deals if d.is_institutional]
        total_buy_cr = sum(d.value_cr for d in deals if d.transaction == "BUY")
        total_sell_cr = sum(d.value_cr for d in deals if d.transaction == "SELL")
        inst_buy_cr = sum(d.value_cr for d in inst_deals if d.transaction == "BUY")
        inst_sell_cr = sum(d.value_cr for d in inst_deals if d.transaction == "SELL")

        result = {
            "total_deals": len(deals),
            "institutional_deals": len(inst_deals),
            "total_buy_cr": round(total_buy_cr, 2),
            "total_sell_cr": round(total_sell_cr, 2),
            "institutional_net_cr": round(inst_buy_cr - inst_sell_cr, 2),
            "deals": [d.to_dict() for d in deals[:20]],
            "note": (
                "Institutional net buying > ₹10 Cr in a single stock is a strong "
                "conviction signal. Promoter buying is even stronger. "
                "FII selling at scale warrants caution."
            ),
        }
        self._cycle_cache[cache_key] = result
        return result

    def _tool_options_positioning(self, symbol: str) -> dict[str, Any]:
        cache_key = f"options_{symbol}"
        if cache_key in self._cycle_cache:
            return self._cycle_cache[cache_key]

        snap = self._options_client.get_snapshot(symbol)
        if snap is None:
            result = {
                "symbol": symbol,
                "available": False,
                "note": "Options data unavailable — check if stock has F&O segment.",
            }
        else:
            result = {"available": True, **snap.to_dict()}
            self._cycle_cache[f"_options_obj_{symbol}"] = snap

        self._cycle_cache[cache_key] = result
        return result

    def _tool_stock_technicals(self, symbol: str) -> dict[str, Any]:
        cache_key = f"technicals_{symbol}"
        if cache_key in self._cycle_cache:
            return self._cycle_cache[cache_key]

        try:
            quote = self._market_data.get_quote(symbol)
            # Daily history for liquidity profile and technical snapshot
            daily_history = self._market_data.get_history(symbol, period="3mo", interval="1d")
            snap = compute_technical_snapshot(daily_history)
            liq = compute_liquidity_profile(
                symbol,
                daily_history,
                quote.price,
                self._settings.max_trade_amount,
                intraday_mode=self._settings.intraday_mode,
            )
            # In intraday mode, also fetch 5-min bars for ATR-based stop placement
            if self._settings.intraday_mode:
                try:
                    intraday_history = self._market_data.get_intraday(symbol, interval="5m")
                    self._cycle_cache[f"_history_{symbol}"] = intraday_history
                except Exception:
                    self._cycle_cache[f"_history_{symbol}"] = daily_history
            else:
                self._cycle_cache[f"_history_{symbol}"] = daily_history
            self._cycle_cache[f"_technical_obj_{symbol}"] = snap
            self._cycle_cache[f"_liquidity_obj_{symbol}"] = liq

            result = {
                "symbol": symbol,
                "quote": {
                    "price": quote.price,
                    "change_pct": quote.change_pct,
                    "open": quote.open,
                    "high": quote.high,
                    "low": quote.low,
                },
                "technicals": snap.to_dict(),
                "liquidity": liq.to_dict(),
            }
        except Exception as exc:
            result = {"symbol": symbol, "error": str(exc)}

        self._cycle_cache[cache_key] = result
        return result

    def _tool_score_opportunity(self, symbol: str) -> dict[str, Any]:
        # Pull from cycle cache where available
        events_data = self._cycle_cache.get(f"events_{symbol}", {})
        flows_data = self._cycle_cache.get(f"flows_{symbol}", {})

        # Re-fetch events/deals if not cached this cycle
        events = self._event_scraper.get_announcements(symbol=symbol, new_only=False)
        deals = self._event_scraper.get_deals_for_symbol(symbol)
        options = self._cycle_cache.get(f"_options_obj_{symbol}")

        liq = self._cycle_cache.get(f"_liquidity_obj_{symbol}")
        adv_cr = liq.avg_daily_value_cr if liq is not None else None
        evidence = self._scorer.score(
            symbol,
            events,
            deals,
            options,
            adv_cr=adv_cr,
            intraday_mode=self._settings.intraday_mode,
            options_staleness_minutes=(
                self._settings.intraday_options_staleness_minutes
                if self._settings.intraday_mode
                else 5.0
            ),
        )
        # Cache for use by execute_trade
        self._cycle_cache[f"_evidence_{symbol}"] = evidence

        return {
            "symbol": symbol,
            **evidence.to_dict(),
        }

    def _tool_portfolio_status(self) -> dict[str, Any]:
        return {
            "portfolio": self._portfolio.summary(),
            "max_positions": self._settings.max_open_positions,
            "dry_run": self._settings.dry_run,
            "capacity_remaining": (
                self._settings.max_open_positions - self._portfolio.position_count
            ),
        }

    def _tool_execute_trade(self, inputs: dict[str, Any]) -> dict[str, Any]:
        symbol = inputs["symbol"].upper()
        quantity = int(inputs["quantity"])
        entry_price = float(inputs["entry_price"])
        stop_loss = float(inputs["stop_loss"])
        take_profit = float(inputs["take_profit"])
        thesis = str(inputs.get("thesis", ""))
        holding_days = int(inputs.get("holding_period_days", 3))

        # Resolve sector for concentration cap
        try:
            profile = self._company_graph.get_profile(symbol)
            sector = profile.sector if profile else None
        except Exception:
            sector = None

        # Portfolio capacity (includes sector concentration check)
        if not self._portfolio.can_open(symbol, sector=sector):
            return {
                "status": "rejected",
                "reason": "Duplicate position, max positions reached, or sector cap hit.",
            }

        # Validate R:R (minimum 2:1)
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0:
            return {"status": "rejected", "reason": "Stop-loss must be below entry price."}
        if reward / risk < 2.0:
            return {
                "status": "rejected",
                "reason": f"Risk:reward {reward/risk:.2f}:1 is below 2:1 minimum.",
            }

        # Liquidity check from cache
        liq = self._cycle_cache.get(f"_liquidity_obj_{symbol}")
        if liq is not None and not liq.is_liquid:
            return {
                "status": "rejected",
                "reason": f"Illiquid stock: tier={liq.liquidity_tier}, ADV=₹{liq.avg_daily_value_cr:.1f}Cr",
            }

        # Slippage — verify stop is not inside the noise/cost zone
        if liq is not None:
            slippage = compute_slippage(entry_price, quantity, liq, stop_loss)
            if stop_loss > entry_price - (entry_price * slippage.break_even_pct * 1.5):
                return {
                    "status": "rejected",
                    "reason": (
                        f"Stop-loss ₹{stop_loss} is inside the break-even cost zone "
                        f"({slippage.break_even_pct*100:.2f}%). "
                        f"Widen stop to at least ₹{slippage.adjusted_stop_loss:.2f}."
                    ),
                }

        entry_order = self._executor.place_market_order(
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            current_price=entry_price,
            notes=thesis[:200],
        )
        sl_order = self._executor.place_stop_loss_order(
            symbol=symbol,
            quantity=quantity,
            trigger_price=stop_loss,
            notes=f"SL for {symbol}",
        )
        self._portfolio.open_position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order=entry_order,
            sl_order=sl_order,
            sector=sector,
        )

        return {
            "status": "executed",
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "holding_period_days": holding_days,
            "dry_run": self._settings.dry_run,
            "entry_order_id": entry_order.order_id,
            "sl_order_id": sl_order.order_id if sl_order else None,
            "thesis": thesis,
        }

    def _tool_close_position(self, symbol: str, reason: str) -> dict[str, Any]:
        symbol = symbol.upper()
        if not self._portfolio.has_position(symbol):
            return {"status": "no_position", "symbol": symbol}
        try:
            price = self._market_data.get_quote(symbol).price
        except Exception:
            pos = self._portfolio.open_positions[symbol]
            price = pos.entry_price

        pos = self._portfolio.open_positions[symbol]
        self._executor.place_market_order(
            symbol, "SELL", pos.quantity, price, notes=f"Close: {reason}"
        )
        trade = self._portfolio.close_position(symbol, price, reason)
        return {
            "status": "closed",
            "symbol": symbol,
            "exit_price": price,
            "pnl": trade.pnl if trade else None,
            "reason": reason,
        }
