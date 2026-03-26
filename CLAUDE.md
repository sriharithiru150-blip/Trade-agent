# CLAUDE.md — Trade-agent

This file documents the codebase structure, development conventions, and workflows for AI assistants (and human developers) working in this repository.

> **Status:** Repository is in initial setup. This CLAUDE.md was created as the first commit to establish conventions before code is added.

---

## Project Overview

**Trade-agent** is an AI-powered trading agent system. The project aims to automate trading decisions, portfolio management, or market analysis using agent-based architectures.

---

## Repository Layout (Planned)

```
Trade-agent/
├── CLAUDE.md               # This file — AI assistant guidance
├── README.md               # Human-facing project documentation
├── .env.example            # Template for required environment variables
├── .gitignore
├── pyproject.toml          # Python project config (or package.json for Node)
│
├── src/                    # Primary source code
│   └── trade_agent/
│       ├── __init__.py
│       ├── agent.py        # Core agent logic
│       ├── strategies/     # Trading strategies
│       ├── data/           # Market data fetchers and models
│       ├── execution/      # Order execution layer
│       └── utils/          # Shared utilities
│
├── tests/                  # Test suite (mirrors src/ structure)
│   ├── unit/
│   └── integration/
│
├── scripts/                # One-off utility scripts, not part of the package
├── docs/                   # Extended documentation
└── .github/
    └── workflows/          # CI/CD pipelines
```

> Update this section as the actual structure is established.

---

## Development Setup

### Prerequisites

- Python 3.11+ (or Node.js 20+ if applicable)
- Git

### First-Time Setup

```bash
# Clone the repository
git clone <repo-url>
cd Trade-agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"     # or: npm install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

### Running the Agent

```bash
# (Update this once entry point is defined)
python -m trade_agent
```

---

## Environment Variables

Document all required and optional environment variables here. Never commit real values to the repository.

| Variable | Required | Description |
|---|---|---|
| `EXCHANGE_API_KEY` | Yes | API key for the trading exchange |
| `EXCHANGE_API_SECRET` | Yes | API secret for the trading exchange |
| `LOG_LEVEL` | No | Logging verbosity (default: `INFO`) |
| `DRY_RUN` | No | If `true`, simulate trades without real execution (default: `false`) |

> Update this table as new variables are added. Always add new variables to `.env.example` at the same time.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/unit/test_agent.py

# Run only fast unit tests (exclude integration)
pytest -m "not integration"
```

### Test Conventions

- Unit tests go in `tests/unit/`, integration tests in `tests/integration/`.
- Test files are named `test_<module>.py` mirroring the source path.
- Use `pytest` fixtures; avoid `unittest.TestCase` classes.
- Mock external APIs and exchange calls in unit tests — never hit live services.
- Integration tests may use sandboxed/paper-trading endpoints; mark them with `@pytest.mark.integration`.

---

## Code Conventions

### Language & Style

- **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/). Use `ruff` for linting and `black` for formatting.
- **Type hints**: Required on all public functions and class methods.
- **Docstrings**: Required on all public modules, classes, and functions (Google style).

### Formatting & Linting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

Run all checks before committing:

```bash
black src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest
```

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Files/modules | `snake_case` | `order_executor.py` |
| Classes | `PascalCase` | `TradeAgent` |
| Functions/variables | `snake_case` | `fetch_market_data()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_POSITION_SIZE` |
| Private members | `_leading_underscore` | `_internal_state` |

### Git Conventions

- **Branch naming**: `feature/<short-description>`, `fix/<issue-or-description>`, `claude/<task-description>`
- **Commit messages**: Imperative mood, concise summary line (≤72 chars), body for context if needed.
  - Good: `Add momentum strategy for BTC/USD pair`
  - Bad: `updated stuff`, `fixes`
- **Never commit**: `.env`, API keys, secrets, large data files, model weights.

---

## Architecture Notes

### Agent Design

The trade agent follows an **observe → decide → act** loop:

1. **Observe**: Fetch market data (prices, order book, indicators) from exchange APIs.
2. **Decide**: Run strategy logic to generate signals (buy/sell/hold).
3. **Act**: Execute orders through the execution layer; log results.

### Key Abstractions (to be implemented)

- `BaseStrategy`: Interface all trading strategies must implement.
- `MarketDataClient`: Abstraction over exchange data feeds (live and historical).
- `OrderExecutor`: Handles order placement, cancellation, and status tracking.
- `PortfolioManager`: Tracks positions, P&L, and risk limits.

### Financial Safety Rules

- Always implement a **kill switch** / circuit breaker to halt trading.
- All order sizes must be validated against position limits before execution.
- Log every trade decision with full context for auditability.
- `DRY_RUN=true` must disable all real order placement without exception.

---

## CI/CD

> Update this section when GitHub Actions workflows are added.

Planned pipeline stages:
1. **Lint & format check** (`ruff`, `black --check`)
2. **Type check** (`mypy`)
3. **Unit tests** (`pytest -m "not integration"`)
4. **Integration tests** (on merge to `main` only)

---

## Security Considerations

- Never log raw API keys, secrets, or account credentials.
- Use environment variables or a secrets manager — no hardcoded credentials.
- Validate and sanitize all external data (market feeds, webhooks) before use.
- Regularly rotate API keys; prefer read-only keys where possible.
- Keep dependencies up to date; use `pip audit` or `safety` to scan for vulnerabilities.

---

## For AI Assistants

When working in this repository:

1. **Read before editing**: Always read existing files before modifying them.
2. **Minimal changes**: Make only the changes needed for the task; don't refactor unrelated code.
3. **No secrets**: Never write API keys, tokens, or passwords into any file.
4. **Financial safety first**: Any code touching order execution must be conservative — prefer failing safely over executing a bad trade.
5. **Tests required**: New logic must have corresponding tests. Do not delete existing tests.
6. **Update `.env.example`**: When adding a new environment variable, always add it to `.env.example` with a description.
7. **Update this file**: When the project structure changes significantly, update the layout section in this CLAUDE.md.
8. **Branch**: Work on the designated feature branch; never push directly to `main`.
