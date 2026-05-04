// BacktestTab — single-config backtest

const MOCK_BT_RESULTS = {
  net_pnl: 7240,
  total_trades: 54,
  win_rate: 0.608,
  sharpe: 2.11,
  sortino: 3.04,
  profit_factor: 1.78,
  max_drawdown: 1800,
  largest_win: 960,
  largest_loss: -620,
};

const MOCK_BT_PARAMS = [
  { label: 'stop_fib',           value: '0.85' },
  { label: 'take_profit_r',      value: '2.50' },
  { label: 'direction',          value: 'Both' },
  { label: 'place_orders_time',  value: '18:05' },
  { label: 'cancel_orders_time', value: '12:00' },
  { label: 'use_risk_sizing',    value: 'True' },
];

const MOCK_SAVED_BTS = [
  'BAC — OOS validation — 2026-04-10 09:15:22',
  'BAC — Entry sweep best — 2026-04-08 14:33:07',
  'BAC-OOS — Baseline — 2026-04-05 11:20:44',
];

function BacktestTab({ strategy, oosRows }) {
  const [selectedBt, setSelectedBt] = React.useState(MOCK_SAVED_BTS[0]);
  const [btLabel, setBtLabel] = React.useState('');
  const [isRunning, setIsRunning] = React.useState(false);
  const [hasResult, setHasResult] = React.useState(true);
  const [loadedFrom, setLoadedFrom] = React.useState('Loaded: BAC — OOS validation');

  const handleRun = () => {
    setIsRunning(true);
    setTimeout(() => { setIsRunning(false); setHasResult(true); }, 1800);
  };

  const fmtDollar = v => v >= 0 ? `$${v.toLocaleString()}` : `-$${Math.abs(v).toLocaleString()}`;
  const fmtPct = v => (v * 100).toFixed(1) + '%';
  const fmtFloat = v => v.toFixed(2);

  const perf = MOCK_BT_RESULTS;

  return (
    <div style={btStyles.root}>
      {/* Saved runs */}
      <div style={btStyles.section}>
        <div style={btStyles.sectionHead}>Saved Backtests</div>
        <div style={{display:'flex',gap:8,alignItems:'center'}}>
          <select style={btStyles.runSelect} value={selectedBt} onChange={e => { setSelectedBt(e.target.value); setLoadedFrom('Loaded: ' + e.target.value.split(' — ').slice(0,2).join(' — ')); }}>
            {MOCK_SAVED_BTS.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <button style={btStyles.btnDanger}>Delete</button>
        </div>
        {loadedFrom && <div style={btStyles.loadedBanner}>📋 {loadedFrom}</div>}
      </div>

      {/* Load from OOS */}
      <div style={btStyles.section}>
        <div style={btStyles.sectionHead}>Load from OOS Results</div>
        <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
          {(oosRows || MOCK_OOS_ROWS).map((r, i) => (
            <button key={i} style={btStyles.oosBtn} onClick={() => setLoadedFrom(`Loaded: OOS config #${i+1}`)}>
              OOS #{i+1} — Sharpe {r.oos_sharpe?.toFixed(2) || '—'}
            </button>
          ))}
        </div>
      </div>

      <div style={btStyles.divider}></div>

      {/* Config */}
      <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
        <div style={{flex:1,minWidth:300}}>
          <div style={btStyles.sectionHead}>Parameters</div>
          <div style={btStyles.paramGrid}>
            {MOCK_BT_PARAMS.map(p => (
              <div key={p.label} style={btStyles.paramRow}>
                <span style={btStyles.paramKey}>{p.label}</span>
                <span style={btStyles.paramVal}>{p.value}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{display:'flex',flexDirection:'column',gap:12,minWidth:240}}>
          <div>
            <div style={btStyles.sectionHead}>Date Range</div>
            <div style={{display:'flex',gap:8}}>
              <div style={{flex:1}}>
                <div style={btStyles.fieldLabel}>Start</div>
                <input style={btStyles.input} defaultValue="2025-03-01" />
              </div>
              <div style={{flex:1}}>
                <div style={btStyles.fieldLabel}>End</div>
                <input style={btStyles.input} defaultValue="2026-01-01" />
              </div>
            </div>
          </div>
          <div>
            <div style={btStyles.fieldLabel}>Backtest label</div>
            <input style={btStyles.input} value={btLabel} onChange={e => setBtLabel(e.target.value)} placeholder="Optional description…" />
          </div>
          <button style={btStyles.btnPrimary} onClick={handleRun} disabled={isRunning}>
            {isRunning ? '⏳ Running…' : '▶ Run Backtest'}
          </button>
        </div>
      </div>

      {/* Results */}
      {hasResult && (
        <>
          <div style={btStyles.divider}></div>
          <div style={btStyles.sectionHead}>Performance</div>
          <div style={btStyles.metricsGrid}>
            <MetricCard label="Net P&L"       value={fmtDollar(perf.net_pnl)}       trend="pos" size="default" />
            <MetricCard label="Sharpe"        value={fmtFloat(perf.sharpe)}          accent="teal" />
            <MetricCard label="Win Rate"      value={fmtPct(perf.win_rate)}          accent="amber" />
            <MetricCard label="Profit Factor" value={fmtFloat(perf.profit_factor)}   accent="teal" />
            <MetricCard label="Max Drawdown"  value={fmtDollar(-perf.max_drawdown)}  trend="neg" />
            <MetricCard label="Total Trades"  value={perf.total_trades.toString()}   />
            <MetricCard label="Largest Win"   value={fmtDollar(perf.largest_win)}    trend="pos" />
            <MetricCard label="Largest Loss"  value={fmtDollar(perf.largest_loss)}   trend="neg" />
          </div>

          {/* Mock equity curve */}
          <div style={btStyles.chartWrap}>
            <div style={btStyles.chartLabel}>Equity Curve</div>
            <svg viewBox="0 0 700 120" style={{width:'100%',height:120}}>
              <defs>
                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(78% 0.17 145)" stopOpacity="0.3"/>
                  <stop offset="100%" stopColor="oklch(78% 0.17 145)" stopOpacity="0"/>
                </linearGradient>
              </defs>
              <path d="M0,100 L30,92 L70,85 L110,78 L140,72 L180,68 L220,74 L260,62 L300,55 L340,48 L380,42 L420,36 L460,44 L500,30 L540,24 L580,18 L620,14 L660,10 L700,8" fill="url(#eq)" stroke="none"/>
              <path d="M0,100 L30,92 L70,85 L110,78 L140,72 L180,68 L220,74 L260,62 L300,55 L340,48 L380,42 L420,36 L460,44 L500,30 L540,24 L580,18 L620,14 L660,10 L700,8" fill="none" stroke="oklch(78% 0.17 145)" strokeWidth="1.5"/>
            </svg>
          </div>

          <div style={btStyles.actionRow}>
            <button style={btStyles.btnGhost}>⬇ Export results (txt)</button>
            <button style={btStyles.btnSecondary}>📋 Load params into Configure &amp; Run</button>
          </div>
        </>
      )}
    </div>
  );
}

const MOCK_OOS_ROWS = [
  { oos_sharpe: 2.11 }, { oos_sharpe: 1.94 }, { oos_sharpe: 1.78 },
];

const btStyles = {
  root: { padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14, fontFamily: "'Inter', sans-serif" },
  section: { display: 'flex', flexDirection: 'column', gap: 8 },
  sectionHead: { fontSize: 11, fontWeight: 600, letterSpacing: '0.09em', textTransform: 'uppercase', color: 'oklch(44% 0.012 240)', paddingBottom: 4, borderBottom: '1px solid oklch(26% 0.02 240)', marginBottom: 2 },
  divider: { height: 1, background: 'oklch(26% 0.02 240)' },
  runSelect: { flex: 1, appearance: 'none', WebkitAppearance: 'none', background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 4, padding: '6px 10px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)' },
  loadedBanner: { fontSize: 12, color: 'oklch(72% 0.145 195)', padding: '5px 10px', background: 'oklch(72% 0.145 195 / 0.08)', border: '1px solid oklch(72% 0.145 195 / 0.25)', borderRadius: 4 },
  oosBtn: { padding: '5px 12px', background: 'oklch(65% 0.155 250 / 0.12)', color: 'oklch(72% 0.14 250)', border: '1px solid oklch(65% 0.155 250 / 0.3)', borderRadius: 4, fontSize: 12, fontFamily: "'JetBrains Mono', monospace", cursor: 'pointer' },
  paramGrid: { display: 'flex', flexDirection: 'column', gap: 4, background: 'oklch(16% 0.016 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 6, overflow: 'hidden' },
  paramRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 12px', borderBottom: '1px solid oklch(26% 0.02 240 / 0.5)' },
  paramKey: { fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(65% 0.012 240)' },
  paramVal: { fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(78% 0.18 75)', fontWeight: 500 },
  fieldLabel: { fontSize: 11, fontWeight: 500, color: 'oklch(65% 0.012 240)', marginBottom: 4 },
  input: { background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 5, padding: '7px 10px', fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)', outline: 'none', width: '100%' },
  metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 },
  chartWrap: { background: 'oklch(16% 0.016 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 7, padding: '10px 14px', overflow: 'hidden' },
  chartLabel: { fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'oklch(44% 0.012 240)', marginBottom: 6 },
  actionRow: { display: 'flex', gap: 8 },
  btnPrimary: { padding: '9px 20px', background: 'oklch(78% 0.18 75)', color: 'oklch(10% 0.01 240)', border: 'none', borderRadius: 5, fontSize: 14, fontWeight: 700, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
  btnSecondary: { padding: '6px 12px', background: 'oklch(20% 0.018 240)', color: 'oklch(94% 0.005 240)', border: '1px solid oklch(32% 0.022 240)', borderRadius: 5, fontSize: 12, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
  btnGhost: { padding: '6px 12px', background: 'transparent', color: 'oklch(65% 0.012 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 5, fontSize: 12, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
  btnDanger: { padding: '6px 10px', background: 'oklch(60% 0.22 25 / 0.1)', color: 'oklch(70% 0.2 25)', border: '1px solid oklch(60% 0.22 25 / 0.35)', borderRadius: 4, fontSize: 12, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
};

Object.assign(window, { BacktestTab, btStyles });
