# Strategy Platform — Dashboard Revamp Design

**Date:** 2026-04-09
**Status:** Draft
**Scope:** Full redesign of `strategy_platform/dashboard/app.py` (3,579 lines)

---

## 1. Problem Summary

Issues from the user's audit grouped into five themes:

### A. Navigation / Flow
- No logical progression through the tabs. The current order (`Configure & Run → In-Sample → Monte Carlo → OOS Validation → Compare Runs → Backtest → Autoresearch`) mixes configuration, results viewing, and single-run testing without clear separation.
- IS, MC, OOS, and Compare Runs are all part of the optimization pipeline output — they belong together, not interleaved with unrelated tabs.

### B. Naming / Labeling
- No way to distinguish optimization runs from backtest runs at a glance in dropdowns and tables.
- Renaming is scattered — it exists in Compare Runs but not at point-of-run in Configure & Run.
- "Combined shortlisted configs" is jargon; should be "Top 5" / "Top 10".
- Compare Runs doesn't clearly state whether you're viewing optimization or backtest data.
- `trade_count` / `total_trades` missing from key tables (Combined + Shortlisted).

### C. Redundancy / Overcomplexity
- Compare Runs tab has too many repeated tables, promote sections, and options showing the same information in slightly different views.
- In-Sample parameter filters are deletable multiselect tiles — confusing because this is a read-only results view, not a configuration screen.
- Backtest saved runs are hidden behind an expander that should be permanently visible.

### D. Bugs
- **Backtest:** Largest win/loss not populated in performance table.
- **Config & Run:** "Include in sweep" checkbox resets when switching param groups (session_state keyed incorrectly).
- **Config & Run:** Combination count doesn't aggregate across all included param groups — only shows combos for the currently visible group.
- **OOS tab:** Latest run sometimes missing from the dropdown until dashboard restart (caching / stale `_run_tss`).
- **IS/OOS:** Boolean "Optimize" mode sweeps dependent params even when the controlling bool is False (e.g. `use_htf=False` still tries all HTF timeframes — wasted compute).
- **Config & Run:** RR target step resolution too coarse (0.5 instead of 0.1).

### E. UX Friction
- **Backtest:** Parameter selectors take up the whole page with no grouping; all params shown flat with no delineation.
- **Backtest:** Selecting a saved run doesn't auto-populate — requires pressing "Load".
- **Backtest:** No way to delete saved backtest results.
- **Backtest:** No params table shown alongside performance results.
- **Config & Run:** No way to name a run at launch time.
- **Config & Run:** Restore button misaligned.
- **Config & Run:** No way to sweep ORB/trading time windows (session times locked to single-select).
- **IS tab:** "Top 50" table should be scrollable (10 rows visible at a time), not full-height.
- **IS tab:** Parameter filters use deletable pills — users don't expect to delete result filters.

---

## 2. Structural Diagnosis

Many of the issues above are symptoms of two deeper problems:

### Problem 1: Flat tab bar conflating three distinct activities

The dashboard has three activities with different user intent:
1. **Configure + launch** an optimization run
2. **View results** of completed optimization runs (IS → MC → OOS → Compare)
3. **Test a single config** against any date range (Backtest)

Currently all seven tabs sit at the same level, making navigation feel random. The fix is a three-zone tab structure with results viewing as a nested sub-navigation.

### Problem 2: No run-type taxonomy

Optimization runs and backtest runs share no labeling convention. The user has to remember what each timestamp means. The fix is an auto-prefix naming system applied at creation time.

---

## 3. Proposed Tab Structure

```
┌─────────────────┐  ┌─────────────────────────────┐  ┌───────────┐  ┌──────────────┐
│ Configure & Run  │  │ Results                      │  │ Backtest   │  │ Autoresearch  │
└─────────────────┘  │  ┌────┬────┬─────┬─────────┐ │  └───────────┘  └──────────────┘
                     │  │ IS │ MC │ OOS │ Compare  │ │
                     │  └────┴────┴─────┴─────────┘ │
                     └─────────────────────────────────┘
```

**4 top-level tabs:**
1. **Configure & Run** — setup, param grids, launch optimization
2. **Results** — read-only views of optimization output, with internal IS / MC / OOS / Compare sub-tabs (implemented via `st.tabs` nested inside the Results tab)
3. **Backtest** — single-config testing on any date range
4. **Autoresearch** — LLM-driven parameter search (unchanged structurally)

**User flow:**
Configure & Run → (wait for pipeline) → Results → (pick best config) → Backtest → (validate on unseen data)

---

## 4. Detailed Spec Per Tab

### 4.1 Configure & Run

**Purpose:** Set up and launch an optimization run.

**What changes from current:**

| Area | Current | New |
|------|---------|-----|
| Run naming | Not possible at launch | Text input "Run label" at top, pre-populated with strategy name. Auto-prefix `OPT_` added to the timestamp label. |
| Restore button | Misaligned (rl2 column too narrow) | Move to same row as the dropdown using `st.columns([5, 1.5])` or a form submit button. |
| Include in sweep | Resets when switching param groups | Persist in `st.session_state` keyed as `inc_{strategy}_{group}` — already keyed this way but value is being lost because the checkbox re-renders with `_prefs_cache.get()` fallback. Fix: only use prefs as initial default, never overwrite live session_state. |
| Sweep state summary | Not visible | Add a colored summary bar below the Parameter Group dropdown: one chip per group showing `Group Name: ✓ included (N combos)` or `✗ excluded`. Total combo count aggregates ALL included groups. |
| Combo count | Shows only current group's combos | Show per-group combo counts in the summary bar AND a bold total at the bottom aggregating all included groups. |
| Time param sweeping | Single-select dropdown | Add a checkbox "Sweep time range" per time param. When checked, replace the single dropdown with From/To pickers (same pattern as numeric params). This enables ORB and session window optimization. |
| RR step resolution | Hardcoded 0.5 | Change `_infer_step()` to respect the param_grid values. If the grid provides `[0.5, 1.0, 1.5, ...]` the step is 0.5. But the Step field in the UI should allow the user to manually type a finer step (e.g. 0.1). The `number_input` for Step should have `min_value` of `0.01` for float params. |
| Custom presets | Not supported | When user manually adjusts Step to a non-standard value, auto-save that step to prefs so it becomes the default next time. |
| Boolean dependent params | Pipeline runs all combos | See Section 6 (Pipeline Fix). UI addition: when a bool param is set to "Optimize", show a caption below dependent params: "Skipped when {bool_param} = False". |

**Controls (in order):**
1. Run label text input (optional, prepopulated with strategy name)
2. Restore from saved run (dropdown + Restore button, same row)
3. Data Settings (start, end, IS split slider)
4. Pipeline Settings (expander, unchanged)
5. Sweep State Summary bar (new)
6. Parameter Group dropdown (if strategy has groups)
7. Include in sweep checkbox
8. Parameter inputs for selected group
9. Total combination count (aggregated across all included groups)
10. Run / Stop buttons

**Data it loads:** `list_run_timestamps()` for the restore dropdown, `_load_prefs()` for saved defaults.

### 4.2 Results (with IS / MC / OOS / Compare sub-tabs)

**Purpose:** Read-only exploration of optimization run output.

**Structural change:** This is a single top-level tab containing `st.tabs(["In-Sample", "Monte Carlo", "OOS Validation", "Compare Runs"])` internally.

**Shared header (above the sub-tabs):**
- Run selector dropdown (shared across all sub-tabs — selecting a run here changes IS, MC, and OOS simultaneously)
- Run label display + inline edit
- Meta banner (data range, IS/OOS cutoff)

This eliminates the duplicated run selector that currently appears independently in IS, MC, and OOS tabs (which can get out of sync).

#### 4.2.1 In-Sample sub-tab

**What changes:**

| Area | Current | New |
|------|---------|-----|
| Parameter filters | Deletable multiselect pills | Standard multiselect (non-deletable). Remove the pill-deletion behavior — these are result filters, not configuration. Use `st.multiselect` with `label_visibility="visible"`. |
| Top 50 table | Full height, all rows visible | Capped at 10 visible rows with vertical scroll. Use `height=df_height(10)` instead of `df_height(len(top50))`. Add a "Show N" dropdown: 10, 25, 50. |
| Trade count | Present in table | Ensure `total_trades` is always in `priority_cols` and is one of the first columns after param keys. |
| Load best → Optimizer | Button exists | Keep as-is. |

**Controls:** Parameter filters, Day-of-week filter, Quick filters (profitable only, positive sharpe), "Show N" dropdown for table size, Export button.

#### 4.2.2 Monte Carlo sub-tab

**What changes:** Minimal. Add `total_trades` to the table's priority_cols. Rename columns to user-friendly labels using `_humanize_columns()`.

#### 4.2.3 OOS Validation sub-tab

**What changes:**

| Area | Current | New |
|------|---------|-----|
| Run dropdown | Sometimes missing latest run | Fix: clear `st.cache_data` for `_list_stage_runs_with_rows` after a pipeline completes. Add `st.cache_data.clear()` call when the pipeline process finishes (detected in the `_output_panel` fragment). |
| Load into Backtester | Exists | Add auto-prefix `OOS_` to the backtest label when loading from OOS. |

#### 4.2.4 Compare Runs sub-tab

**What changes:**

| Area | Current | New |
|------|---------|-----|
| Complexity | 4 separate sections (Best Per Run, Promote Run Winner, Combined Shortlisted, Promote Shortlisted, Param Frequency, Head-to-Head) all expanded | Collapse to 3 sections: **Best Per Run** (always visible), **Top N Pool** (expander, renamed from "Combined shortlisted configs"), **Parameter Stability** (expander, renamed from "Parameter Frequency"). Remove the separate "Promote A Shortlisted Config" section — merge promote buttons into the Top N Pool table as action buttons per row. |
| "Combined shortlisted" name | Jargon | Rename to "Top {N} Pool" where N comes from the Top N / run setting. |
| Trade count | Missing from tables | Add `total_trades` to `best_cols` and `shortlist_cols`. |
| Run type clarity | Ambiguous | Add a clear banner: "Comparing **optimization** runs at the **{stage}** stage". |
| Head-to-Head by Entry Mode | Only works for strategies with `entry_mode` | Make it generic: if `entry_mode` doesn't exist, offer a dropdown to pick any categorical param for the grouping dimension. If no categorical params exist, hide the section entirely. |
| Promote sections | Two separate promote sections (run winner + shortlisted) | Single promote section at the bottom. Dropdown selects any row from either Best Per Run or Top N Pool. Two buttons: "Send to Backtest" / "Send to Optimizer". |

### 4.3 Backtest

**Purpose:** Run a single parameter configuration against any date range.

**What changes:**

| Area | Current | New |
|------|---------|-----|
| Saved runs | Hidden behind expander | Always visible as a dropdown at the top. Selecting a run auto-populates params, dates, and performance (no separate "Load" button). |
| Delete runs | Not possible | Add a "Delete" button next to the saved run dropdown (with confirmation). |
| Parameter UI | Flat grid of all params, no grouping, takes up the whole page | Group params using the strategy's `param_groups` (same groups as Configure & Run). Show as a dropdown to select the group, then display only that group's params. Each param is a `selectbox` from `param_grid` values with an adjacent `number_input` for custom override. If the strategy has no `param_groups`, render params in a compact 4-column grid with clear labels. |
| Custom values | Only grid values available | Each param has a small "Custom" toggle. When enabled, replaces the selectbox with a `number_input` (numeric) or `text_input` (categorical) so the user can type any value. |
| Performance table | Shows metrics but not the params that produced them | Add a "Parameters" table below the performance metrics showing every param and its value for this run. Use the same `_config_details_df()` helper. |
| Largest win/loss | Bug: not populated | Fix: ensure the strategy's `run_backtest()` returns `largest_win` and `largest_loss` in the result dict. If missing, compute from the trades DataFrame: `trades['pnl'].max()` and `trades['pnl'].min()`. Add a fallback in `_render_nt_metrics()`. |
| Auto-prefix | No type distinction | All saved backtests auto-prefixed with `BAC_` in their label. When loaded from OOS, prefix is `BAC_OOS_`. |

**Layout (top to bottom):**
1. Saved runs dropdown (always visible, auto-loads on selection) + Delete button
2. Load from OOS results (if available)
3. Run label text input
4. Info banner: "Loaded: {source}"
5. Parameter group dropdown → param inputs for selected group (compact)
6. Data source radio (MySQL / NT CSV)
7. Date range inputs
8. Run Backtest button
9. --- (divider) ---
10. Performance metrics table (NT-style, two columns)
11. Parameters table (all params and their values)
12. Equity curve, Monthly P&L, Day-of-Week, Trade List (unchanged)

### 4.4 Autoresearch

**No structural changes.** Remains its own top-level tab.

Minor fixes:
- Add `total_trades` to the generation log table if not already present.
- Ensure run selector shows all saved runs without requiring a restart.

---

## 5. Naming Convention

### Auto-prefix system

Every saved result gets an automatic prefix based on its type:

| Type | Prefix | Example label |
|------|--------|---------------|
| Optimization run | `OPT` | `OPT — 2026-04-09 14:30` |
| Backtest (manual) | `BAC` | `BAC — 2026-04-09 15:00` |
| Backtest loaded from OOS | `BAC-OOS` | `BAC-OOS — 2026-04-09 15:10` |
| Autoresearch run | `AR` | `AR — 2026-04-09 16:00` |

### User-appended labels

The user can optionally append a description. The system preserves the prefix + timestamp and appends the user's text:

```
OPT — Entry architecture sweep — 2026-04-09 14:30
BAC-OOS — Baseline validation — 2026-04-09 15:10
```

### Where labels appear

- All run selector dropdowns
- Compare Runs table (`_run_display` column)
- Backtest saved runs list

### Implementation

- **Optimization runs:** Add a `run_label` text input in Configure & Run. When the pipeline launches, prepend `OPT` to the label. Store via `results_store.set_run_label()`.
- **Backtests:** When `_save_backtest()` is called, auto-prepend `BAC` (or `BAC-OOS` if loaded from OOS tab). Store via `results_store.set_backtest_label()`.
- **Display:** Modify `_format_run_display()` and `_fmt_bt_file()` to always show: `{prefix} — {user_label} — {formatted_timestamp}`.

---

## 6. Bug Fixes

### BUG-1: Largest win/loss not populated in Backtest

**Root cause:** The strategy's `run_backtest()` may not return `largest_win` / `largest_loss` keys in the result dict.

**Fix:** In the Backtest tab results rendering (around line 3036), add a fallback:
```python
trades_df = result.get("trades")
if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
    if "largest_win" not in result or pd.isna(result.get("largest_win")):
        result["largest_win"] = trades_df["pnl"].max()
    if "largest_loss" not in result or pd.isna(result.get("largest_loss")):
        result["largest_loss"] = trades_df["pnl"].min()
```

### BUG-2: "Include in sweep" resets when switching param groups

**Root cause:** The checkbox widget key `inc_{strategy}_{group}` is correct, but when the group dropdown changes, Streamlit re-renders and the `if _inc_key not in st.session_state` block re-initializes from `_prefs_cache` (which may be False).

**Fix:** Change the initialization logic:
```python
# Only set from prefs on first-ever render, never overwrite existing session_state
if _inc_key not in st.session_state:
    st.session_state[_inc_key] = _prefs_cache.get(_inc_key, False)
```
The current code already does this, so the real bug is that `_apply_pending_optimizer_include_state()` or some other path is clearing the key. Trace all `st.session_state.pop()` calls that touch `inc_` keys and ensure they only clear the CURRENT group's key during a reset, not all groups.

Additionally, verify that switching the `selgrp_{strategy}` selectbox does not trigger a full re-render path that clears other groups' include state. The fix is to ensure all `inc_` keys are preserved across group switches — only clear them on explicit "Reset grid" actions.

### BUG-3: Combo count doesn't aggregate across param groups

**Root cause:** `custom_grid` only gets populated for the currently visible group (via `_render_group_params`) and the loop at line 1799 that populates from other groups' session_state. But if a group was included but its params were never rendered in this session, the session_state keys `run_grid_{key}` may not exist.

**Fix:** In the combo count section (line 1831), iterate ALL groups (not just `custom_grid`). For each included group, pull values from `st.session_state[f"run_grid_{key}"]` falling back to `param_grid[key]`. Compute the product across all included groups.

### BUG-4: OOS tab missing latest run

**Root cause:** `_list_stage_runs_with_rows()` is decorated with `@st.cache_data`, so it returns stale results until the cache is invalidated.

**Fix:** After the pipeline subprocess completes (detected in the `_output_panel` fragment when `st.session_state.run_done` becomes True), call:
```python
_list_stage_runs_with_rows.clear()
load_run_csv.clear()
_load_and_label.clear()
```
This forces all result-loading caches to refresh on the next render.

### BUG-5: Boolean optimization waste

**Root cause:** When `use_htf_confirmation` is set to "Optimize" (sweep True + False), the pipeline runs every combination of `htf_timeframe` for BOTH True and False. When False, the HTF timeframe is irrelevant — all combos produce identical results.

**Fix location:** `strategy_platform/optimize/pipeline.py`, in the grid combination generator.

**Fix approach:** Add a `param_dependencies` property to `BaseStrategy` (optional, default empty dict):
```python
@property
def param_dependencies(self) -> Dict[str, tuple]:
    """Map of param -> (controlling_bool, required_value).
    When the controlling bool != required_value, collapse this param to its first grid value."""
    return {}
```

In `_grid_combinations()` (or a wrapper), before generating combos:
1. Read `strategy.param_dependencies`.
2. For each combo, check if any dependent param's controller is set to the non-required value.
3. If so, fix the dependent param to its first value and deduplicate.

This is the same concept as `param_conditional` in the dashboard but applied at the pipeline level. Strategies that already define `param_conditional` can reuse the same mapping.

### BUG-6: RR target step too coarse

**Root cause:** `_infer_step()` computes step from the most common gap in the param_grid values. If the grid is `[0.5, 1.0, 1.5, 2.0]`, the inferred step is 0.5.

**Fix:** The inferred step is correct as a default, but the Step `number_input` should allow the user to type a finer value. Change `min_value` for float params from `float(_step_v) * 0.01` to `0.01` (hardcoded minimum). This lets users set step=0.1 even if the grid's natural step is 0.5.

---

## 7. Implementation Order

Phases are ordered by impact and dependency. Each phase is self-contained and testable.

### Phase 1: Bug fixes (no UI restructure)

1. **BUG-1:** Largest win/loss fallback in Backtest tab
2. **BUG-4:** Cache clearing after pipeline completion (OOS stale dropdown)
3. **BUG-6:** RR step resolution — change min_value to 0.01
4. **BUG-2:** Include in sweep state persistence — trace and fix session_state clearing
5. **BUG-3:** Combo count aggregation across all included groups

**Files:** `app.py` only (except BUG-5 which is Phase 2).

### Phase 2: Pipeline boolean optimization fix (BUG-5)

1. Add `param_dependencies` property to `BaseStrategy` (default empty)
2. Implement combo deduplication in `pipeline.py`
3. Add `param_dependencies` to existing strategies that have `param_conditional` (ORB15M, WickTest5M, etc.)
4. Add caption in Configure & Run UI showing which params will be collapsed

**Files:** `base_strategy.py`, `pipeline.py`, each strategy's `strategy.py`, `app.py`.

### Phase 3: Naming convention

1. Add `run_label` text input to Configure & Run
2. Implement auto-prefix system (`OPT`, `BAC`, `BAC-OOS`, `AR`)
3. Update `_format_run_display()` and `_fmt_bt_file()` to show prefix + label + timestamp
4. Update `_save_backtest()` to auto-prefix based on source

**Files:** `app.py`, `results_store.py` (label format only).

### Phase 4: Tab restructure

1. Replace 7-tab layout with 4-tab layout
2. Move IS, MC, OOS, Compare into nested `st.tabs()` inside the Results tab
3. Add shared run selector header above the sub-tabs
4. Remove per-sub-tab run selectors
5. Reorder: Configure & Run → Results → Backtest → Autoresearch

**Files:** `app.py` (major restructure — this is the largest phase).

### Phase 5: Backtest tab UX overhaul

1. Move saved runs to always-visible dropdown with auto-load on selection
2. Add delete button with confirmation dialog
3. Restructure param UI: group dropdown → group params (compact), with custom value toggle
4. Add params table below performance metrics
5. Auto-prefix `BAC` / `BAC-OOS` on save

**Files:** `app.py`.

### Phase 6: Configure & Run UX improvements

1. Add sweep state summary bar (chips per group showing include status + combo count)
2. Add time-param sweep mode (From/To pickers with checkbox toggle)
3. Fix Restore button alignment
4. Persist custom step values to prefs

**Files:** `app.py`.

### Phase 7: Results tab polish

1. IS: Replace deletable multiselect pills with standard multiselect
2. IS: Add "Show N" dropdown for Top 50 table (10/25/50 rows)
3. Compare: Simplify to 3 sections (Best Per Run, Top N Pool, Parameter Stability)
4. Compare: Add `total_trades` to all tables
5. Compare: Rename "Combined shortlisted configs" → "Top N Pool"
6. Compare: Merge two promote sections into one
7. Compare: Make Head-to-Head grouping generic (any categorical param, not just entry_mode)
8. Compare: Add "Comparing optimization runs at the {stage} stage" banner

**Files:** `app.py`.

---

## 8. Risk Notes

- **Phase 4 (tab restructure)** is the highest-risk change. It touches the entire layout and all cross-tab state management. Test thoroughly with at least 2 different strategies.
- **Phase 2 (pipeline bool fix)** changes optimization output. Existing saved results won't retroactively benefit, but new runs will be more efficient. No data migration needed.
- **Phase 5 (Backtest UX)** changes how params are rendered. Ensure the new grouped layout works for strategies with AND without `param_groups`.
- The shared run selector in Phase 4 means IS/MC/OOS always show the same run. If a user wants to compare IS from run A with OOS from run B, they'll need to use the Compare tab. This is a deliberate simplification — the current independent selectors create confusion about what you're looking at.
