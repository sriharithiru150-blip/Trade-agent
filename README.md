# Trade-agent

An AI-powered day-trading agent for NSE and BSE markets, driven by **Claude (Anthropic)** as the reasoning engine.

The agent continuously monitors a watchlist of Indian equities, aggregates market data and news (including from subsidiaries, partners, and competitors), runs technical analysis, generates sentiment scores, and executes — or paper-trades — intraday positions automatically.

---

## Features

| Capability | Details |
|---|---|
| **Market data** | Real-time quotes and OHLCV history via `yfinance` (NSE `.NS` / BSE `.BO`) |
| **News intelligence** | NewsAPI + Google News RSS; searches company, subsidiaries, partners, and competitors |
| **AI reasoning** | Claude claude-sonnet-4-6 via tool-use to analyse, score, and decide trades |
| **Technical analysis** | RSI, MACD, Bollinger Bands, Volume, ATR via `ta` library |
| **Sentiment scoring** | Claude scores news articles on relevance and market impact |
| **Combined signals** | Weighted blend of technical + sentiment + market-breadth signals |
| **Order execution** | Zerodha Kite Connect (live) or built-in paper-trade engine |
| **Risk management** | Per-trade stop-loss, take-profit, max positions, kill switch |
| **Scheduling** | Runs on a configurable interval during NSE/BSE market hours (IST) |

---

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd Trade-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install (with dev extras)
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY

# 5. Run in paper-trade mode (safe default)
python -m trade_agent
```

---

## Configuration

All settings are in `.env` (copy from `.env.example`). Key variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `NEWS_API_KEY` | — | NewsAPI key for news fetching |
| `DRY_RUN` | `true` | Paper-trade mode — **always start here** |
| `WATCHLIST` | RELIANCE,TCS,… | Comma-separated NSE symbols |
| `MAX_TRADE_AMOUNT` | `10000` | Max INR per trade |
| `STOP_LOSS_PCT` | `0.02` | 2% stop-loss per position |
| `TAKE_PROFIT_PCT` | `0.04` | 4% take-profit target |
| `KILL_SWITCH` | `false` | Set to `true` to halt all trading immediately |
| `MIN_SIGNAL_SCORE` | `0.65` | Minimum combined score (0–1) to open a trade |
| `ANALYSIS_INTERVAL_MINUTES` | `30` | How often the agent re-analyses |

---

## Architecture

```
Observe → Decide → Act  (runs every N minutes during market hours)
```

```
trade_agent/
├── agent.py            # Main Claude-powered orchestrator (tool-use loop)
├── config.py           # Pydantic settings from environment
├── data/
│   ├── market_data.py  # yfinance wrapper — quotes, OHLCV
│   ├── news_fetcher.py # NewsAPI + RSS feeds; entity-aware search
│   └── company_graph.py# Company → subsidiary/partner/competitor relationships
├── analysis/
│   ├── technical.py    # RSI, MACD, BB, Volume, ATR
│   ├── sentiment.py    # Claude-based news sentiment scorer
│   └── signals.py      # Weighted signal aggregation
├── strategies/
│   ├── base.py         # Abstract BaseStrategy
│   ├── trend_following.py
│   └── news_driven.py
├── execution/
│   ├── order_executor.py    # Kite Connect + paper-trade engine
│   └── portfolio_manager.py # Positions, P&L, risk checks
└── utils/
    ├── logger.py        # Structured logging (structlog)
    └── market_hours.py  # IST market hours helpers
```

---

## Safety

- `DRY_RUN=true` **must** be set in `.env` for paper trading (default).
- `KILL_SWITCH=true` immediately halts all order placement — use in emergencies.
- Every trade decision is logged with full context for auditability.
- Position sizes are validated against `MAX_TRADE_AMOUNT` and `MAX_OPEN_POSITIONS` before any order.
- Stop-loss orders are placed immediately after every entry.

---

## Development

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Tests
pytest --cov=src --cov-report=term-missing

# All checks (run before committing)
black src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest
```

---

## Disclaimer

This software is for **educational and research purposes only**. Automated trading carries significant financial risk. The authors are not responsible for any losses incurred. Always test thoroughly in paper-trade mode before enabling live trading.
