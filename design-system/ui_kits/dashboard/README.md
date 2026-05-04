# Strategy Platform — Dashboard UI Kit

## Overview
High-fidelity click-through prototype of the redesigned Strategy Platform dashboard.
Covers the full 4-tab layout: Configure & Run, Results (IS/MC/OOS/Compare), Backtest, and Autoresearch.

## Design System
- **Colors**: Dark terminal aesthetic — charcoal backgrounds, amber primary accent, teal secondary, monochrome neutrals
- **Type**: Space Grotesk (headings) · Inter (UI) · JetBrains Mono (data/numbers)
- **Icons**: Emoji inline (⚙️ 📈 🔬 🔁 ▶ ⬇) — production version should use Lucide Icons from CDN

## Files
| File | Description |
|---|---|
| `index.html` | Main prototype entry point — full dashboard with sidebar + tabs |
| `Sidebar.jsx` | Strategy selector, bar type, symbol info |
| `MetricCard.jsx` | KPI card component (large, default, mini sizes) |
| `DataTable.jsx` | Sortable results table with monospace numbers, profit/loss color, badges |
| `ConfigureRun.jsx` | Configure & Run tab — params, sweep state, run button, progress |
| `ResultsTab.jsx` | Results tab with IS/MC/OOS/Compare sub-tabs |
| `BacktestTab.jsx` | Backtest tab — saved runs, OOS load, params, performance, equity curve |

## Usage
Open `index.html` in a browser. All data is mocked — no backend needed.

Click through:
1. Select a strategy in the sidebar (goldbot7, mobobands, orb15m, etc.)
2. Configure parameters in the ⚙️ Configure & Run tab — click **▶ Run Optimization**
3. Switch to 📈 Results → explore IS / MC / OOS / Compare sub-tabs
4. Load a config into 🔬 Backtest — run backtest, view equity curve
5. Try 🔁 Autoresearch — click Start to see generation log animate

## Design Width
1440px wide (full-screen terminal app). No responsive breakpoints — this is a desktop tool.
