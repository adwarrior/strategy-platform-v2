// ResultsTab — Results tab with IS / MC / OOS / Compare sub-tabs

const MOCK_IS_ROWS = [
  { rank: 1, config: 'stop_fib=0.85  tp_r=2.50  dir=Both',        net_pnl: 18240, sharpe: 2.84, win_rate: 0.632, trades: 142, profit_factor: 1.94, max_dd: -2640, status: 'Shortlisted' },
  { rank: 2, config: 'stop_fib=0.90  tp_r=2.00  dir=Both',        net_pnl: 16800, sharpe: 2.61, win_rate: 0.614, trades: 138, profit_factor: 1.82, max_dd: -2900, status: 'Shortlisted' },
  { rank: 3, config: 'stop_fib=0.80  tp_r=3.00  dir=Long Only',   net_pnl: 14200, sharpe: 2.44, win_rate: 0.581, trades: 119, profit_factor: 1.71, max_dd: -3100, status: 'Shortlisted' },
  { rank: 4, config: 'stop_fib=0.75  tp_r=2.50  dir=Both',        net_pnl: 12400, sharpe: 2.12, win_rate: 0.553, trades: 131, profit_factor: 1.58, max_dd: -3600, status: null },
  { rank: 5, config: 'stop_fib=0.70  tp_r=2.00  dir=Both',        net_pnl:  9800, sharpe: 1.88, win_rate: 0.541, trades: 144, profit_factor: 1.42, max_dd: -4100, status: null },
  { rank: 6, config: 'stop_fib=0.95  tp_r=1.50  dir=Short Only',  net_pnl: -2800, sharpe: -0.42, win_rate: 0.412, trades: 88, profit_factor: 0.74, max_dd: -6200, status: 'Rejected' },
];

const MOCK_OOS_ROWS = [
  { rank: 1, config: 'stop_fib=0.85  tp_r=2.50  dir=Both',   oos_net_pnl: 7240, oos_sharpe: 2.11, oos_win_rate: 0.608, oos_trades: 54, oos_max_dd: -1800 },
  { rank: 2, config: 'stop_fib=0.90  tp_r=2.00  dir=Both',   oos_net_pnl: 6200, oos_sharpe: 1.94, oos_win_rate: 0.589, oos_trades: 51, oos_max_dd: -2100 },
  { rank: 3, config: 'stop_fib=0.80  tp_r=3.00  dir=Long Only', oos_net_pnl: 5400, oos_sharpe: 1.78, oos_win_rate: 0.571, oos_trades: 44, oos_max_dd: -2400 },
];

function ResultsTab({ runTss, strategy }) {
  const [subTab, setSubTab] = React.useState('is');
  const [selectedRun, setSelectedRun] = React.useState(runTss[0] || '');
  const [selectedRow, setSelectedRow] = React.useState(0);
  const [showDeleteConfirm, setShowDeleteConfirm] = React.useState(false);

  const formatTs = ts => ts ? `OPT — ${ts.replace(/_(\d{4})$/, ' $1').replace(/_/g, '-')}` : '—';

  return (
    <div style={rtStyles.root}>
      {/* Shared run selector */}
      <div style={rtStyles.runBar}>
        <div style={rtStyles.runBarLeft}>
          <div style={rtStyles.runLabel}>View results from run</div>
          <select style={rtStyles.runSelect} value={selectedRun} onChange={e => setSelectedRun(e.target.value)}>
            {runTss.slice(0,6).map(ts => <option key={ts} value={ts}>{formatTs(ts)}</option>)}
          </select>
        </div>
        <div style={rtStyles.runBarRight}>
          <div style={rtStyles.metaBadge}>
            <span style={rtStyles.metaLabel}>IS</span>
            <span style={rtStyles.metaVal}>2023-01-01 → 2025-03-01</span>
          </div>
          <div style={rtStyles.metaBadge}>
            <span style={rtStyles.metaLabel}>OOS</span>
            <span style={rtStyles.metaVal}>2025-03-01 → 2026-01-01</span>
          </div>
          <button style={rtStyles.btnDanger} onClick={() => setShowDeleteConfirm(!showDeleteConfirm)}>Delete run</button>
        </div>
      </div>
      {showDeleteConfirm && (
        <div style={rtStyles.deleteWarn}>
          ⚠️ Permanently delete this run — IS, MC, and OOS results?
          <button style={rtStyles.btnDangerSolid} onClick={() => setShowDeleteConfirm(false)}>Yes, delete</button>
          <button style={rtStyles.btnGhost} onClick={() => setShowDeleteConfirm(false)}>Cancel</button>
        </div>
      )}

      {/* Sub-tab pills */}
      <div style={rtStyles.subTabBar}>
        {[{id:'is',label:'In-Sample'},{id:'mc',label:'Monte Carlo'},{id:'oos',label:'OOS Validation'},{id:'cmp',label:'Compare Runs'}].map(t => (
          <div key={t.id} style={{...rtStyles.subTab, ...(subTab===t.id ? rtStyles.subTabActive : {})}} onClick={() => setSubTab(t.id)}>{t.label}</div>
        ))}
      </div>

      {/* Sub-tab content */}
      {subTab === 'is' && <ISSubTab rows={MOCK_IS_ROWS} selectedRow={selectedRow} onRowSelect={setSelectedRow} />}
      {subTab === 'mc' && <MCSubTab />}
      {subTab === 'oos' && <OOSSubTab rows={MOCK_OOS_ROWS} selectedRow={selectedRow} onRowSelect={setSelectedRow} />}
      {subTab === 'cmp' && <CompareSubTab runTss={runTss} />}
    </div>
  );
}

function ISSubTab({ rows, selectedRow, onRowSelect }) {
  const cols = [
    { key: 'rank',          label: '#',             numeric: true },
    { key: 'config',        label: 'Configuration', type: 'code' },
    { key: 'net_pnl',       label: 'Net P&L',       numeric: true, type: 'dollar', colored: true },
    { key: 'sharpe',        label: 'Sharpe',        numeric: true, type: 'float', colored: true },
    { key: 'win_rate',      label: 'Win Rate',      numeric: true, type: 'pct' },
    { key: 'trades',        label: 'Trades',        numeric: true, type: 'int' },
    { key: 'profit_factor', label: 'PF',            numeric: true, type: 'float', colored: true },
    { key: 'max_dd',        label: 'Max DD',        numeric: true, type: 'dollar', colored: true },
    { key: 'status',        label: 'Status',        type: 'badge' },
  ];
  const sel = rows[selectedRow];
  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <DataTable columns={cols} rows={rows} selectedRow={selectedRow} onRowClick={(i) => onRowSelect(i)} />
      {sel && (
        <div style={rtStyles.actionRow}>
          <div style={rtStyles.selConfig}>{sel.config}</div>
          <button style={rtStyles.btnSecondary}>📋 Load into Configure &amp; Run</button>
          <button style={rtStyles.btnSecondary}>🔬 Send to Backtest</button>
          <button style={rtStyles.btnGhost}>⬇ Export IS Results (txt)</button>
        </div>
      )}
    </div>
  );
}

function MCSubTab() {
  const mcRows = [
    { config: 'stop_fib=0.85  tp_r=2.50', mc_stability: 0.87, mc_sharpe_p5: 1.12, mc_pnl_p50: 14800, mc_pnl_p5: 8200, net_pnl: 18240 },
    { config: 'stop_fib=0.90  tp_r=2.00', mc_stability: 0.81, mc_sharpe_p5: 0.98, mc_pnl_p50: 13200, mc_pnl_p5: 6800, net_pnl: 16800 },
    { config: 'stop_fib=0.80  tp_r=3.00', mc_stability: 0.76, mc_sharpe_p5: 0.88, mc_pnl_p50: 11400, mc_pnl_p5: 5400, net_pnl: 14200 },
  ];
  const cols = [
    { key: 'config',        label: 'Configuration', type: 'code' },
    { key: 'mc_stability',  label: 'MC Stability',  numeric: true, type: 'pct', colored: true },
    { key: 'mc_sharpe_p5',  label: 'Sharpe P5',     numeric: true, type: 'float', colored: true },
    { key: 'mc_pnl_p50',    label: 'P&L P50',       numeric: true, type: 'dollar', colored: true },
    { key: 'mc_pnl_p5',     label: 'P&L P5',        numeric: true, type: 'dollar', colored: true },
    { key: 'net_pnl',       label: 'IS Net P&L',    numeric: true, type: 'dollar', colored: true },
  ];
  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <div style={rtStyles.infoBar}>200 simulations · Day-shuffle Monte Carlo · IS period only</div>
      <DataTable columns={cols} rows={mcRows} />
    </div>
  );
}

function OOSSubTab({ rows, selectedRow, onRowSelect }) {
  const cols = [
    { key: 'rank',          label: '#',             numeric: true },
    { key: 'config',        label: 'Configuration', type: 'code' },
    { key: 'oos_net_pnl',   label: 'OOS Net P&L',  numeric: true, type: 'dollar', colored: true },
    { key: 'oos_sharpe',    label: 'OOS Sharpe',    numeric: true, type: 'float', colored: true },
    { key: 'oos_win_rate',  label: 'OOS Win Rate',  numeric: true, type: 'pct' },
    { key: 'oos_trades',    label: 'Trades',        numeric: true, type: 'int' },
    { key: 'oos_max_dd',    label: 'OOS Max DD',    numeric: true, type: 'dollar', colored: true },
  ];
  const sel = rows[selectedRow];
  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <DataTable columns={cols} rows={rows} selectedRow={selectedRow} onRowClick={(i) => onRowSelect(i)} />
      {sel && (
        <div style={rtStyles.actionRow}>
          <div style={rtStyles.selConfig}>{sel.config}</div>
          <button style={rtStyles.btnSecondary}>📋 Load OOS config #{selectedRow+1} into Backtester</button>
          <button style={rtStyles.btnSecondary}>🔒 Lock params into Configure &amp; Run</button>
        </div>
      )}
    </div>
  );
}

function CompareSubTab({ runTss }) {
  const bestRows = runTss.slice(0,3).map((ts, i) => ({
    run: `OPT — ${ts.slice(0,8).replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3')}`,
    config: i === 0 ? 'stop_fib=0.85  tp_r=2.50' : i === 1 ? 'stop_fib=0.90  tp_r=2.00' : 'stop_fib=0.80  tp_r=3.00',
    net_pnl: [18240, 15600, 12800][i],
    sharpe:  [2.84, 2.41, 2.09][i],
    win_rate:[0.632, 0.601, 0.581][i],
    trades:  [142, 128, 119][i],
  }));
  const cols = [
    { key: 'run',      label: 'Run',      type: 'code' },
    { key: 'config',   label: 'Best Config', type: 'code' },
    { key: 'net_pnl',  label: 'Net P&L',  numeric: true, type: 'dollar', colored: true },
    { key: 'sharpe',   label: 'Sharpe',   numeric: true, type: 'float', colored: true },
    { key: 'win_rate', label: 'Win Rate', numeric: true, type: 'pct' },
    { key: 'trades',   label: 'Trades',   numeric: true, type: 'int' },
  ];
  return (
    <div style={{display:'flex',flexDirection:'column',gap:14}}>
      <div style={rtStyles.infoBar}>Comparing IS optimization runs · {runTss.length} runs selected</div>
      <div style={rtStyles.compareSectionHead}>Best Per Run</div>
      <DataTable columns={cols} rows={bestRows} />
      <div style={rtStyles.actionRow}>
        <button style={rtStyles.btnSecondary}>📋 Send to Backtester</button>
        <button style={rtStyles.btnSecondary}>📋 Send to Configure &amp; Run</button>
      </div>
    </div>
  );
}

const rtStyles = {
  root: { padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: 14, fontFamily: "'Inter', sans-serif" },
  runBar: { display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '10px 14px', background: 'oklch(16% 0.016 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 7 },
  runBarLeft: { display: 'flex', alignItems: 'center', gap: 8, flex: 1 },
  runBarRight: { display: 'flex', alignItems: 'center', gap: 8 },
  runLabel: { fontSize: 11, color: 'oklch(65% 0.012 240)', fontWeight: 500, whiteSpace: 'nowrap' },
  runSelect: { appearance: 'none', WebkitAppearance: 'none', background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 4, padding: '5px 22px 5px 8px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)', flex: 1, minWidth: 280 },
  metaBadge: { display: 'flex', alignItems: 'center', gap: 5, padding: '3px 8px', background: 'oklch(20% 0.018 240)', borderRadius: 4, border: '1px solid oklch(26% 0.02 240)' },
  metaLabel: { fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'oklch(65% 0.012 240)' },
  metaVal: { fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)' },
  deleteWarn: { display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: 'oklch(60% 0.22 25 / 0.1)', border: '1px solid oklch(60% 0.22 25 / 0.35)', borderRadius: 5, fontSize: 12, color: 'oklch(70% 0.2 25)' },
  subTabBar: { display: 'flex', gap: 2, background: 'oklch(16% 0.016 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 6, padding: 3, width: 'fit-content' },
  subTab: { padding: '5px 16px', fontSize: 12, fontWeight: 500, borderRadius: 4, color: 'oklch(65% 0.012 240)', cursor: 'pointer', transition: 'all 100ms', whiteSpace: 'nowrap' },
  subTabActive: { background: 'oklch(20% 0.018 240)', color: 'oklch(94% 0.005 240)', boxShadow: '0 1px 3px rgba(0,0,0,0.4)' },
  actionRow: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  selConfig: { fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(72% 0.145 195)', flex: 1 },
  infoBar: { fontSize: 12, color: 'oklch(65% 0.012 240)', padding: '6px 10px', background: 'oklch(16% 0.016 240)', borderRadius: 4, border: '1px solid oklch(26% 0.02 240)' },
  compareSectionHead: { fontSize: 12, fontWeight: 700, color: 'oklch(94% 0.005 240)', letterSpacing: '0.04em', textTransform: 'uppercase' },
  btnSecondary: { padding: '6px 12px', background: 'oklch(20% 0.018 240)', color: 'oklch(94% 0.005 240)', border: '1px solid oklch(32% 0.022 240)', borderRadius: 5, fontSize: 12, fontWeight: 500, fontFamily: "'Inter', sans-serif", cursor: 'pointer', whiteSpace: 'nowrap' },
  btnGhost: { padding: '6px 12px', background: 'transparent', color: 'oklch(65% 0.012 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 5, fontSize: 12, fontFamily: "'Inter', sans-serif", cursor: 'pointer', whiteSpace: 'nowrap' },
  btnDanger: { padding: '5px 10px', background: 'oklch(60% 0.22 25 / 0.1)', color: 'oklch(70% 0.2 25)', border: '1px solid oklch(60% 0.22 25 / 0.35)', borderRadius: 4, fontSize: 11, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
  btnDangerSolid: { padding: '5px 12px', background: 'oklch(60% 0.22 25)', color: 'white', border: 'none', borderRadius: 4, fontSize: 12, fontFamily: "'Inter', sans-serif", cursor: 'pointer', fontWeight: 600 },
};

Object.assign(window, { ResultsTab, ISSubTab, MCSubTab, OOSSubTab, CompareSubTab, rtStyles });
