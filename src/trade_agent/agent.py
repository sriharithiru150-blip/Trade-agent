"""Main AI trading agent — Claude-powered orchestrator.

The agent runs an **observe → decide → act** loop:

1. **Observe**: Fetch market data + news for each watchlist symbol and the
   broader market.
2. **Decide**: Hand all gathered data to Claude (via tool-use) which reasons
   across technical indicators, news sentiment, and market context to produce
   a final recommendation.
3. **Act**: Execute the recommended trades through the execution layer,
   subject to portfolio risk limits.

Claude is given a set of tools that map to the data/analysis pipeline.
The agent calls Claude with those tools, Claude calls them, and the agent
executes the results — a tight human-like reasoning loop.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from trade_agent.analysis.sentiment import SentimentAnalyser
from trade_agent.analysis.signals import compute_signal
from trade_agent.analysis.technical import compute_technical_snapshot
from trade_agent.config import Settings
from trade_agent.data.company_graph import CompanyGraph
from trade_agent.data.market_data import MarketDataClient
from trade_agent.data.news_fetcher import NewsFetcher
from trade_agent.execution.order_executor import KillSwitchError, OrderExecutor
from trade_agent.execution.portfolio_manager import PortfolioManager
from trade_agent.strategies.news_driven import NewsDrivenStrategy
from trade_agent.strategies.trend_following import TrendFollowingStrategy
from trade_agent.utils.logger import get_logger
from trade_agent.utils.market_hours import is_market_open, now_ist

log = get_logger(__name__)

# ── Tool definitions for Claude ──────────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_market_overview",
        "description": (
            "Get a snapshot of the broader Indian market: Nifty 50, Sensex, "
            "Bank Nifty, and India VIX. Returns the day's open, current price, "
            "and percentage change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_data",
        "description": (
            "Fetch the current quote and last 3-month daily OHLCV history for "
            "an NSE-listed stock. Also computes technical indicators "
            "(RSI, MACD, Bollinger Bands, ATR, SMA, Volume ratio)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol, e.g. RELIANCE, TCS",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news_and_sentiment",
        "description": (
            "Fetch recent news for a stock from multiple sources (NewsAPI, "
            "Google News RSS, financial portals). Automatically includes news "
            "about subsidiaries, JV partners, key competitors, and suppliers. "
            "Returns scored articles with sentiment and relevance ratings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio_status",
        "description": "Return the current portfolio: open positions, realised/unrealised P&L, and today's trade history.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "execute_trade",
        "description": (
            "Execute a BUY order for a stock. A stop-loss order is automatically "
            "placed after entry. The agent must only call this tool when it has "
            "high conviction. In DRY_RUN mode this is a paper trade — no real "
            "money is at risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "NSE stock symbol"},
                "quantity": {"type": "integer", "description": "Number of shares to buy"},
                "entry_price": {"type": "number", "description": "Expected entry price (market order)"},
                "stop_loss": {"type": "number", "description": "Stop-loss price"},
                "take_profit": {"type": "number", "description": "Take-profit target price"},
                "rationale": {"type": "string", "description": "Brief explanation of the trade decision"},
            },
            "required": ["symbol", "quantity", "entry_price", "stop_loss", "take_profit", "rationale"],
        },
    },
    {
        "name": "close_position",
        "description": "Close an open position at the current market price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "reason": {"type": "string", "description": "Why the position is being closed"},
            },
            "required": ["symbol", "reason"],
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert AI day-trading agent specialising in Indian equity markets (NSE/BSE).

Your objective is to identify high-probability intraday trading opportunities and execute them with disciplined risk management.

## Process
1. First call `get_market_overview` to assess broad market conditions.
2. For each stock in the watchlist, call `get_stock_data` then `get_news_and_sentiment`.
3. Synthesise technical indicators, news sentiment, and market breadth.
4. Call `get_portfolio_status` to check current exposure.
5. Execute trades only when ALL of the following are satisfied:
   - Combined signal score ≥ 0.65
   - At least medium confidence
   - Portfolio has capacity (< max positions)
   - Market is not in extreme bearish mode (Nifty down > 1.5%)
   - No position already open for this symbol

## Risk Rules (MANDATORY)
- Never risk more than the configured max_trade_amount per trade.
- Always specify stop_loss and take_profit when calling execute_trade.
- Minimum risk:reward ratio = 1:2  (stop_loss ≤ 2%, take_profit ≥ 4%).
- If India VIX > 25, reduce position sizing by 50% and tighten stop-losses.
- If Nifty is down > 1.5% intraday, take only high-conviction (score ≥ 0.75) trades.
- Close all positions at least 15 minutes before market close (3:15 PM IST).

## News Analysis Guidelines
- Earnings surprises, guidance changes, and regulatory actions carry HIGH weight.
- Subsidiary/competitor news: analyse the knock-on effect carefully.
- Macro news (RBI, budget, FII flows): weigh against existing technical trend.
- Rumours and unverified reports: reduce confidence level.

## Output
After completing analysis, provide a concise summary:
- Stocks analysed and their signal scores
- Trades executed (or skipped with reason)
- Portfolio status
- Key risks or observations
"""


class TradingAgent:
    """AI-powered trading agent orchestrated by Claude.

    Args:
        settings: Application settings (loaded from environment).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._market_data = MarketDataClient()
        self._news_fetcher = NewsFetcher(news_api_key=settings.news_api_key)
        self._company_graph = CompanyGraph()
        self._sentiment = SentimentAnalyser(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )
        self._executor = OrderExecutor(
            dry_run=settings.dry_run,
            kill_switch=settings.kill_switch,
            kite_api_key=settings.kite_api_key,
            kite_access_token=settings.kite_access_token,
        )
        self._portfolio = PortfolioManager(
            max_open_positions=settings.max_open_positions,
            max_trade_amount=settings.max_trade_amount,
        )
        self._trend_strategy = TrendFollowingStrategy(
            min_score=settings.min_signal_score,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_trade_amount=settings.max_trade_amount,
        )
        self._news_strategy = NewsDrivenStrategy(
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_trade_amount=settings.max_trade_amount,
        )

        log.info(
            "agent_initialised",
            model=settings.claude_model,
            dry_run=settings.dry_run,
            watchlist=settings.get_watchlist(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run_once(self) -> str:
        """Run a single analysis-and-execution cycle.

        Returns:
            The final text summary from Claude.
        """
        if not is_market_open():
            log.info("market_closed", time=now_ist().isoformat())
            return "Market is currently closed. No trades executed."

        log.info("agent_cycle_start", time=now_ist().isoformat())
        watchlist = self._settings.get_watchlist()
        initial_message = (
            f"Run a complete market analysis cycle for the following NSE watchlist: "
            f"{', '.join(watchlist)}.\n\n"
            f"Current time (IST): {now_ist().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"DRY_RUN mode: {self._settings.dry_run}"
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
        final_text = ""

        # Agentic tool-use loop
        for _iteration in range(20):  # safety cap
            response = self._client.messages.create(
                model=self._settings.claude_model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            # Collect any text
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text

            # If Claude is done, break
            if response.stop_reason == "end_turn":
                break

            # Handle tool calls
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

                # Append assistant turn + tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        log.info("agent_cycle_complete", summary_length=len(final_text))
        return final_text

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Route a Claude tool call to the appropriate handler."""
        try:
            if name == "get_market_overview":
                return self._tool_market_overview()
            elif name == "get_stock_data":
                return self._tool_stock_data(inputs["symbol"])
            elif name == "get_news_and_sentiment":
                return self._tool_news_sentiment(inputs["symbol"])
            elif name == "get_portfolio_status":
                return self._tool_portfolio_status()
            elif name == "execute_trade":
                return self._tool_execute_trade(inputs)
            elif name == "close_position":
                return self._tool_close_position(inputs["symbol"], inputs.get("reason", "agent"))
            else:
                return {"error": f"Unknown tool: {name}"}
        except KillSwitchError as exc:
            return {"error": f"KILL SWITCH ACTIVE: {exc}"}
        except Exception as exc:
            log.error("tool_error", tool=name, error=str(exc))
            return {"error": str(exc)}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_market_overview(self) -> dict[str, Any]:
        """Fetch broad market indices."""
        from trade_agent.data.market_data import INDEX_TICKERS
        import yfinance as yf

        overview: dict[str, Any] = {}
        for name, ticker in INDEX_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = float(info.last_price)
                prev = float(info.previous_close)
                chg_pct = round(((price - prev) / prev) * 100, 3) if prev else 0.0
                overview[name] = {
                    "price": price,
                    "change_pct": chg_pct,
                    "day_high": float(info.day_high),
                    "day_low": float(info.day_low),
                }
            except Exception as exc:
                overview[name] = {"error": str(exc)}
        log.debug("tool_market_overview", indices=list(overview.keys()))
        return {"market_overview": overview}

    def _tool_stock_data(self, symbol: str) -> dict[str, Any]:
        """Fetch quote + technical indicators for a symbol."""
        try:
            quote = self._market_data.get_quote(symbol)
            history = self._market_data.get_history(symbol, period="3mo", interval="1d")
            snap = compute_technical_snapshot(history)
            return {
                "symbol": symbol,
                "quote": {
                    "price": quote.price,
                    "change_pct": quote.change_pct,
                    "open": quote.open,
                    "high": quote.high,
                    "low": quote.low,
                    "volume": quote.volume,
                    "market_cap": quote.market_cap,
                },
                "technicals": snap.to_dict(),
            }
        except Exception as exc:
            return {"symbol": symbol, "error": str(exc)}

    def _tool_news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch news for symbol + related entities, score sentiment."""
        profile = self._company_graph.get_profile(symbol)
        articles = self._news_fetcher.fetch_for_symbol(
            symbol=symbol,
            company_name=profile.full_name,
            related_entities=profile.related_names(),
        )
        sentiment = self._sentiment.analyse(
            symbol=symbol,
            company_name=profile.full_name,
            articles=articles,
        )
        return {
            "symbol": symbol,
            "company": profile.full_name,
            "related_entities_searched": profile.related_names(),
            "sentiment": sentiment.to_dict(),
        }

    def _tool_portfolio_status(self) -> dict[str, Any]:
        """Return current portfolio summary."""
        return {
            "portfolio": self._portfolio.summary(),
            "max_positions": self._settings.max_open_positions,
            "dry_run": self._settings.dry_run,
        }

    def _tool_execute_trade(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a BUY order and place a stop-loss."""
        symbol: str = inputs["symbol"].upper()
        quantity: int = int(inputs["quantity"])
        entry_price: float = float(inputs["entry_price"])
        stop_loss: float = float(inputs["stop_loss"])
        take_profit: float = float(inputs["take_profit"])
        rationale: str = inputs.get("rationale", "")

        # Portfolio capacity check
        if not self._portfolio.can_open(symbol):
            return {
                "status": "rejected",
                "reason": f"Cannot open position for {symbol}: already open or max positions reached.",
            }

        # Validate risk:reward
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 1.5:
            return {
                "status": "rejected",
                "reason": f"Risk:reward ratio {reward/risk:.2f} is below minimum 1.5",
            }

        # Entry order
        entry_order = self._executor.place_market_order(
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            current_price=entry_price,
            notes=rationale[:200],
        )

        # Stop-loss order
        sl_order = self._executor.place_stop_loss_order(
            symbol=symbol,
            quantity=quantity,
            trigger_price=stop_loss,
            notes=f"SL for {symbol}",
        )

        # Register in portfolio
        self._portfolio.open_position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order=entry_order,
            sl_order=sl_order,
        )

        return {
            "status": "executed",
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "dry_run": self._settings.dry_run,
            "entry_order_id": entry_order.order_id,
            "sl_order_id": sl_order.order_id if sl_order else None,
            "rationale": rationale,
        }

    def _tool_close_position(self, symbol: str, reason: str) -> dict[str, Any]:
        """Close an open position at market price."""
        symbol = symbol.upper()
        if not self._portfolio.has_position(symbol):
            return {"status": "no_position", "symbol": symbol}
        try:
            quote = self._market_data.get_quote(symbol)
            current_price = quote.price
        except Exception:
            pos = self._portfolio.open_positions.get(symbol)
            current_price = pos.entry_price if pos else 0.0

        # Sell order
        pos = self._portfolio.open_positions[symbol]
        self._executor.place_market_order(
            symbol=symbol,
            transaction_type="SELL",
            quantity=pos.quantity,
            current_price=current_price,
            notes=f"Close: {reason}",
        )
        trade = self._portfolio.close_position(symbol, current_price, reason)
        return {
            "status": "closed",
            "symbol": symbol,
            "exit_price": current_price,
            "pnl": trade.pnl if trade else None,
            "reason": reason,
        }

    # ── End-of-day square-off ─────────────────────────────────────────────────

    def square_off_all(self) -> dict[str, Any]:
        """Close all open positions — call at end of trading day.

        Returns:
            Summary of closed trades and total P&L.
        """
        prices: dict[str, float] = {}
        for sym in self._portfolio.open_positions:
            try:
                prices[sym] = self._market_data.get_quote(sym).price
            except Exception:
                prices[sym] = self._portfolio.open_positions[sym].entry_price

        # Place SELL orders for each open position
        for sym, pos in self._portfolio.open_positions.items():
            price = prices.get(sym, pos.entry_price)
            self._executor.place_market_order(
                symbol=sym,
                transaction_type="SELL",
                quantity=pos.quantity,
                current_price=price,
                notes="EOD square-off",
            )

        trades = self._portfolio.close_all_positions(prices, reason="eod")
        total_pnl = sum(t.pnl for t in trades)
        log.info("eod_squareoff", trades=len(trades), total_pnl=round(total_pnl, 2))
        return {
            "trades_closed": len(trades),
            "total_pnl": round(total_pnl, 2),
            "trades": [t.to_dict() for t in trades],
        }
