# Strategy Platform Design System

## Overview

**Strategy Platform** is a personal quantitative trading research tool — a Python/Streamlit backtesting and optimization engine for NinjaTrader strategies. It lets the user configure strategy parameter grids, run multi-stage optimization pipelines (In-Sample → Monte Carlo → Out-of-Sample), run single-configuration backtests, and drive autonomous parameter search (Autoresearch). It is a single-user internal tool, not a public product.

**Product surface:** one product — a full-screen Streamlit dashboard (`strategy_platform/dashboard/app.py`, ~4,900 lines).

---

## Sources

| Source | Path / Notes |
|---|---|
| Codebase (attached) | `strategy-platform/` via File System Access API |
| Main dashboard | `strategy-platform/strategy_platform/dashboard/app.py` |
| Design revamp doc | `strategy-platform/docs/revamp-design.md` |
| Strategy examples | `strategy-platform/strategy_platform/strategies/goldbot7/`, `mobobands/`, `orb15m/`, etc. |
| Base strategy contract | `strategy-platform/strategy_platform/base_strategy.py` |
| Sample report heatmaps | `strategy-platform/reports/heatmap_*.png` |

No Figma link was provided.

---

## Product Context

The platform auto-adapts to any registered strategy. The sidebar lets the user pick a strategy (e.g. `goldbot7`, `mobobands`, `orb15m`) and configure bar type (minute vs tick). The main area has **4 top-level tabs**:

1. **⚙️ Configure & Run** — Set date range, IS/OOS split, parameter groups, sweep ranges, then launch pipeline
2. **📈 Results** — Read-only nested sub-tabs: In-Sample / Monte Carlo / OOS Validation / Compare Runs
3. **🔬 Backtest** — Single-config test against any date range; saved runs; load from OOS
4. **🔁 Autoresearch** — LLM-driven autonomous parameter search (hill-climbing)

Strategies: `goldbot7` (Gold futures PDH/PDL breakout), `mobobands` (Mobo Bands DPO strategy on MNQ), `orb15m` (Opening Range Breakout), `wicktest5m`, `patscalp`, `swingstrat`, `waejurikpro`, `nybreakout`.

---

## CONTENT FUNDAMENTALS

### Tone & Voice
- **Direct, technical, no-fluff.** This is a solo developer's internal tool. Copy is functional, not marketing.
- **First-person singular absent** — labels are imperative or descriptive ("Configure & Run", "View results", "Run Backtest"), not "I" or "you".
- **Abbreviations used freely**: IS (In-Sample), OOS (Out-of-Sample), MC (Monte Carlo), P&L, PF (Profit Factor), RR (Risk:Reward), NT (NinjaTrader), Sharpe, Sortino — no need to spell out.
- **Casing**: Title Case for tab names and section headers; Sentence case for captions and help text.
- **Emoji used sparingly** as tab/button icons: ⚙️ Configure, 📈 Results, 🔬 Backtest, 🔁 Autoresearch, ▶ Run, ⬇ Export, ✅ success, ⚠️ warning. Never decorative.
- **Numbers formatted**: `$1,234`, `34.5%`, `2.34` (2dp float), `1,234 combos`. Dollar amounts with `$` prefix, commas.
- **Naming convention**: `OPT — Label — YYYY-MM-DD HH:MM`, `BAC — Label — timestamp`, `AR — ...`
- **Messages**: Concise status captions like `"Switch to the 📈 Results tab to view output."` — always actionable.
- **No excessive punctuation** — no exclamation marks except in ✅ success toasts.

### Specific Copy Examples
- `"Set the date range, enable parameter groups, configure ranges, then click Run."`
- `"No results yet — use the ⚙️ Configure & Run tab to run the optimizer."`
- `"Permanently delete this entire run — IS, MC, and OOS results."`
- `"Slow — evaluates signal on every tick"`
- `"Enter comma-separated integers, e.g. 144, 233, 377"`
- Metric labels: `Net P&L`, `Profit Factor`, `Sharpe`, `Win Rate`, `Max Drawdown`, `MC Stability`

---

## VISUAL FOUNDATIONS

### Colors
The app uses **Streamlit's default dark theme** with no custom CSS injection beyond minimal spacing adjustments. The color palette is:

| Role | Value | Notes |
|---|---|---|
| Background | `#0e1117` | Streamlit dark bg |
| Surface / Card | `#1c1e26` | Streamlit widget bg |
| Border | `#31333f` | Subtle dividers |
| Foreground primary | `#fafafa` | Main text |
| Foreground secondary | `#888` / `#aaa` | Captions, help text |
| Accent / Primary | `#ff4b4b` | Streamlit red — primary buttons |
| Success / Profit | `#2ecc71` | Green — kept improvements, positive PnL |
| Loss / Danger | `#e74c3c` | Red — reverted, negative PnL, errors |
| Info / Chart line | `#3498db` | Blue — running best line, info highlights |
| Warning / Orange | `#f39c12` | Orange — caption warnings |
| Plotly default trace | `#636efa` | Default Plotly blue |

For the **redesign**, the design system targets a slicker dark trading terminal aesthetic: charcoal backgrounds, monospaced data, amber/teal accents over black — inspired by Bloomberg Terminal, TradingView dark, and quantitative research tools.

**Proposed redesign palette:**
| Token | Value | Role |
|---|---|---|
| `--bg-base` | `oklch(12% 0.01 240)` | Page background |
| `--bg-surface` | `oklch(17% 0.015 240)` | Card / panel bg |
| `--bg-elevated` | `oklch(21% 0.018 240)` | Hover surface, input bg |
| `--bg-overlay` | `oklch(24% 0.02 240)` | Tooltip, dropdown |
| `--border-subtle` | `oklch(28% 0.02 240)` | Dividers, card borders |
| `--border-strong` | `oklch(38% 0.02 240)` | Active input borders |
| `--fg-primary` | `oklch(95% 0.005 240)` | Primary text |
| `--fg-secondary` | `oklch(65% 0.01 240)` | Secondary / captions |
| `--fg-muted` | `oklch(45% 0.01 240)` | Placeholder, disabled |
| `--accent-amber` | `oklch(78% 0.18 75)` | Primary accent (run buttons, active tabs) |
| `--accent-teal` | `oklch(72% 0.14 195)` | Secondary accent (charts, links) |
| `--accent-green` | `oklch(68% 0.18 145)` | Profit / positive / success |
| `--accent-red` | `oklch(62% 0.22 25)` | Loss / negative / danger |
| `--accent-blue` | `oklch(65% 0.15 250)` | Info / chart running line |
| `--accent-orange` | `oklch(72% 0.18 60)` | Warning |

### Typography
Streamlit uses system sans-serif by default. The redesign uses:

- **Display / headings**: `"DM Mono"` or `"Space Grotesk"` — technical, geometric
- **Body / labels**: `"Inter"` — clean, legible at small sizes
- **Data / numbers**: `"JetBrains Mono"` — monospaced for metric values, tables, code
- **Scale**: 11px captions → 13px labels → 15px body → 18px subhead → 24px section → 32px title

No Google Fonts files were in the codebase. Substituting from Google Fonts CDN (flagged below in ICONOGRAPHY).

### Spacing & Layout
- **Sidebar**: fixed ~280px, collapsible; contains strategy selector, bar type, symbol picker
- **Content area**: full-width, max ~1400px in wide mode; top-level tabs at page top
- **Column grids**: 2, 3, 4 columns used for param inputs (From/To/Step), metric KPI rows
- **Section rhythm**: `---` dividers between logical sections; expanders for secondary config
- **Density**: medium — not cramped, not airy. Tables show 10 rows by default with scroll

### Background & Surfaces
- **Flat dark surfaces** — no gradients, no imagery, no textures
- Card-like distinction via subtle border (`1px solid var(--border-subtle)`) + slightly lighter bg
- No drop shadows in Streamlit default; redesign can use `box-shadow: 0 1px 3px rgba(0,0,0,0.5)`

### Animation & Interactions
- **No animations** in the current app — Streamlit doesn't support custom transitions
- Hover states: slightly lighter background on interactive elements
- Press states: Streamlit primary button darkens slightly on click (red → darker red)
- For the redesign (HTML UI kit): `transition: background 120ms ease, color 120ms ease` — fast, subtle

### Charts (Plotly)
- Dark background `#0e1117`, grid lines `#31333f`
- Colors: `#2ecc71` (kept/profit), `#e74c3c` (reverted/loss), `#3498db` (running best, line), `#636efa` (default scatter)
- Heatmaps: yellow-green-blue diverging scale for parameter performance grids
- Height: 380px standard; 500px for equity curves

### Cards & Tables
- Streamlit `st.dataframe` with dark theme — no custom card styling in current app
- Redesign: monospace number cells, colored P&L (green/red), sticky header row
- Corner radius: `6px` for cards, `4px` for inputs, `2px` for chips/badges

### Borders & Shadows
- Current: Streamlit defaults (subtle 1px borders on inputs)
- Redesign: `1px solid var(--border-subtle)` on cards; `0 0 0 2px var(--accent-amber)` for active/focused inputs

### Iconography usage
- **Emoji as icons** for tab labels and button prefixes (⚙️ 📈 🔬 🔁 ▶ ⬇ ✅ ⚠️ 📋 🔒)
- No icon font or SVG icon system in the codebase
- Redesign: use **Lucide Icons** (CDN) for a clean stroke-weight icon system

---

## ICONOGRAPHY

The codebase has **no dedicated icon system** — it uses emoji glyphs inline in Streamlit markdown and button labels. No SVG icons, no icon font, no PNG icons.

**Emoji icons used:**
- ⚙️ Configure / settings
- 📈 Results / charts
- 🔬 Backtest / science / precision
- 🔁 Autoresearch / loop
- ▶ Run / play
- ⬇ Download / export
- ✅ Success
- ⚠️ Warning
- 📋 Load / clipboard
- 🔒 Lock params

**Redesign icon system:** Lucide Icons (https://unpkg.com/lucide@latest) — stroke-weight 1.5, 16×16 or 20×20 at UI density. Loaded from CDN. See `ui_kits/dashboard/index.html`.

No logo, no brand mark, no illustrations found in the codebase. `assets/` folder contains only generated report heatmap PNGs (not brand assets).

---

## FILE INDEX

```
README.md                          This file — full design system documentation
SKILL.md                           Agent skill manifest
colors_and_type.css                CSS custom properties: color tokens, type scale, spacing

assets/                            Visual assets (no brand logo found; heatmap examples)

preview/                           Design System tab card previews
  colors-base.html                 Base color palette swatches
  colors-semantic.html             Semantic / state colors
  type-scale.html                  Typography scale specimen
  type-mono.html                   Monospace / data typography
  spacing-tokens.html              Spacing + radius + shadow tokens
  components-buttons.html          Button states
  components-inputs.html           Form input states
  components-metrics.html          KPI metric card components
  components-table.html            Data table component
  components-tabs.html             Tab bar component
  components-badges.html           Status badge / chip components
  components-sidebar.html          Sidebar component

ui_kits/
  dashboard/
    README.md                      UI kit documentation
    index.html                     Full dashboard prototype (click-through)
    Sidebar.jsx                    Sidebar component
    TopNav.jsx                     Top-level tab bar
    ConfigureRun.jsx               Configure & Run tab
    ResultsTab.jsx                 Results tab with sub-tabs
    BacktestTab.jsx                Backtest tab
    MetricCard.jsx                 KPI metric card
    DataTable.jsx                  Sortable data table
```

---

## Font Substitution Notice

⚠️ **No font files were found in the codebase** (Streamlit uses system fonts). The design system uses Google Fonts CDN:
- `Space Grotesk` → display headings
- `Inter` → body / UI labels  
- `JetBrains Mono` → data / numbers / monospace

If you have licensed font files to use instead, add them to `fonts/` and update the `@font-face` declarations in `colors_and_type.css`.
