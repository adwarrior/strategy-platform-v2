// ConfigureRun — Configure & Run tab

const MOCK_PARAMS = {
  goldbot7: [
    { key: 'stop_fib',           label: 'Stop Fib',              type: 'numeric', from: 0.70, to: 1.00, step: 0.05, default: 0.90 },
    { key: 'take_profit_r',      label: 'Take Profit R',         type: 'numeric', from: 1.00, to: 4.00, step: 0.25, default: 2.0 },
    { key: 'direction',          label: 'Direction',             type: 'categorical', options: ['Both','Long Only','Short Only'], default: 'Both' },
    { key: 'place_orders_time',  label: 'Place Orders Time',     type: 'time', options: ['18:05','18:30','19:00'], default: '18:05' },
    { key: 'cancel_orders_time', label: 'Cancel Orders Time',    type: 'time', options: ['10:00','11:00','12:00'], default: '12:00' },
    { key: 'use_risk_sizing',    label: 'Use Risk Sizing',       type: 'bool', default: true },
  ],
  mobobands: [
    { key: 'profit_ticks',  label: 'Profit Ticks',  type: 'numeric', from: 8, to: 24, step: 2, default: 12 },
    { key: 'stop_ticks',    label: 'Stop Ticks',    type: 'numeric', from: 4, to: 12, step: 2, default: 6 },
    { key: 'jma_period',    label: 'JMA Period',    type: 'numeric', from: 8, to: 20, step: 2, default: 14 },
    { key: 'mobo_period',   label: 'Mobo Period',   type: 'numeric', from: 10, to: 30, step: 5, default: 20 },
    { key: 'use_htf',       label: 'Use HTF Filter',type: 'bool', default: false },
  ],
  orb15m: [
    { key: 'profit_ticks',  label: 'Profit Ticks',  type: 'numeric', from: 10, to: 30, step: 5, default: 20 },
    { key: 'stop_ticks',    label: 'Stop Ticks',    type: 'numeric', from: 6, to: 14, step: 2, default: 8 },
    { key: 'direction',     label: 'Direction',     type: 'categorical', options: ['Both','Long Only','Short Only'], default: 'Both' },
    { key: 'orb_minutes',   label: 'ORB Minutes',   type: 'numeric', from: 15, to: 60, step: 15, default: 15 },
  ],
};

function ConfigureRun({ strategy, runTss }) {
  const [runLabel, setRunLabel] = React.useState('');
  const [restoreTs, setRestoreTs] = React.useState(runTss[0] || '');
  const [isRunning, setIsRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [isDone, setIsDone] = React.useState(false);
  const [paramState, setParamState] = React.useState({});
  const [includeGroups, setIncludeGroups] = React.useState({ 'Entry params': true, 'Stop params': true, 'Filter params': false });

  const params = MOCK_PARAMS[strategy] || MOCK_PARAMS.goldbot7;

  React.useEffect(() => { setParamState({}); setIsDone(false); setProgress(0); setIsRunning(false); }, [strategy]);

  const handleRun = () => {
    setIsRunning(true); setIsDone(false); setProgress(0);
    let p = 0;
    const iv = setInterval(() => {
      p += Math.random() * 12 + 3;
      if (p >= 100) { p = 100; clearInterval(iv); setIsRunning(false); setIsDone(true); }
      setProgress(Math.min(p, 100));
    }, 300);
  };

  const totalCombos = params.reduce((acc, p) => {
    if (p.type === 'numeric') {
      const n = Math.round((p.to - p.from) / p.step) + 1;
      return acc * n;
    }
    if (p.type === 'categorical') return acc * p.options.length;
    if (p.type === 'bool') return acc * 2;
    return acc;
  }, 1);

  return (
    <div style={crStyles.root}>
      {/* Header row */}
      <div style={crStyles.row}>
        <div style={crStyles.field}>
          <label style={crStyles.label}>Run label <span style={crStyles.optional}>(optional)</span></label>
          <input
            style={crStyles.input}
            value={runLabel}
            onChange={e => setRunLabel(e.target.value)}
            placeholder={`${strategy} optimization…`}
          />
          <span style={crStyles.caption}>Saved as: OPT — {runLabel || strategy} — 2026-05-03 …</span>
        </div>
        <div style={crStyles.field}>
          <label style={crStyles.label}>Restore from saved run</label>
          <div style={{display:'flex',gap:6}}>
            <select style={{...crStyles.input, flex:1}} value={restoreTs} onChange={e => setRestoreTs(e.target.value)}>
              {runTss.slice(0,6).map(ts => <option key={ts} value={ts}>OPT — {ts.replace(/_/g,' ')}</option>)}
            </select>
            <button style={crStyles.btnSecondary}>Restore</button>
          </div>
        </div>
      </div>

      {/* Data settings */}
      <div style={crStyles.sectionHead}>Data Settings</div>
      <div style={crStyles.row}>
        <div style={crStyles.field}>
          <label style={crStyles.label}>Start date</label>
          <input style={crStyles.input} type="text" defaultValue="2023-01-01" />
        </div>
        <div style={crStyles.field}>
          <label style={crStyles.label}>End date</label>
          <input style={crStyles.input} type="text" defaultValue="2026-01-01" />
        </div>
        <div style={crStyles.field}>
          <label style={crStyles.label}>IS split</label>
          <div style={crStyles.splitRow}>
            <div style={crStyles.splitBar}>
              <div style={crStyles.splitIS}>IS 70%</div>
              <div style={crStyles.splitOOS}>OOS 30%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Sweep state summary */}
      <div style={crStyles.sweepSummary}>
        {Object.entries(includeGroups).map(([g, inc]) => (
          <div key={g} style={{...crStyles.sweepChip, ...(inc ? crStyles.sweepChipOn : crStyles.sweepChipOff)}}
            onClick={() => setIncludeGroups(prev => ({...prev, [g]: !prev[g]}))}>
            <span>{inc ? '✓' : '✗'}</span>
            <span>{g}</span>
          </div>
        ))}
        <div style={crStyles.comboCount}>{totalCombos.toLocaleString()} total combos</div>
      </div>

      {/* Params grid */}
      <div style={crStyles.sectionHead}>Parameters</div>
      <div style={crStyles.paramsGrid}>
        {params.map(p => (
          <div key={p.key} style={crStyles.paramBlock}>
            <div style={crStyles.paramLabel}>{p.label} <span style={crStyles.paramDefault}>default: {String(p.default)}</span></div>
            {p.type === 'numeric' && (
              <div style={crStyles.rangeRow}>
                <div style={crStyles.rangeField}><div style={crStyles.rangeLabel}>From</div><input style={crStyles.rangeInput} defaultValue={p.from} /></div>
                <div style={crStyles.rangeField}><div style={crStyles.rangeLabel}>To</div><input style={crStyles.rangeInput} defaultValue={p.to} /></div>
                <div style={crStyles.rangeField}><div style={crStyles.rangeLabel}>Step</div><input style={crStyles.rangeInput} defaultValue={p.step} /></div>
              </div>
            )}
            {p.type === 'categorical' && (
              <div style={crStyles.catRow}>
                {p.options.map(o => (
                  <label key={o} style={crStyles.catOption}>
                    <input type="checkbox" defaultChecked={o === p.default} style={{accentColor:'oklch(78% 0.18 75)'}} />
                    <span style={{fontSize:12}}>{o}</span>
                  </label>
                ))}
              </div>
            )}
            {p.type === 'bool' && (
              <div style={crStyles.radioGroup}>
                {['True','False','Optimize'].map(o => (
                  <div key={o} style={{...crStyles.radioOpt, ...(o === 'True' ? crStyles.radioOptActive : {})}}>{o}</div>
                ))}
              </div>
            )}
            {p.type === 'time' && (
              <select style={crStyles.input}>
                {p.options.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            )}
          </div>
        ))}
      </div>

      {/* Run bar */}
      <div style={crStyles.runBar}>
        {isRunning ? (
          <button style={crStyles.btnDanger} onClick={() => { setIsRunning(false); }}>■ Stop</button>
        ) : (
          <button style={crStyles.btnPrimary} onClick={handleRun}>▶ Run Optimization</button>
        )}
        {isDone && <div style={crStyles.successMsg}>✅ Pipeline complete — switch to 📈 Results to view output.</div>}
        {isRunning && (
          <div style={crStyles.progressWrap}>
            <div style={crStyles.progressLabel}>Running… {Math.round(progress)}%</div>
            <div style={crStyles.progressBar}><div style={{...crStyles.progressFill, width: `${progress}%`}}></div></div>
          </div>
        )}
      </div>
    </div>
  );
}

const crStyles = {
  root: { padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16, fontFamily: "'Inter', sans-serif" },
  row: { display: 'flex', gap: 16, flexWrap: 'wrap' },
  field: { display: 'flex', flexDirection: 'column', gap: 5, flex: 1, minWidth: 200 },
  label: { fontSize: 12, fontWeight: 500, color: 'oklch(65% 0.012 240)', letterSpacing: '0.02em' },
  optional: { fontSize: 11, color: 'oklch(44% 0.012 240)', fontWeight: 400 },
  caption: { fontSize: 11, color: 'oklch(44% 0.012 240)', fontFamily: "'JetBrains Mono', monospace" },
  input: { background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 5, padding: '7px 10px', fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)', outline: 'none', width: '100%' },
  sectionHead: { fontSize: 11, fontWeight: 600, letterSpacing: '0.09em', textTransform: 'uppercase', color: 'oklch(44% 0.012 240)', paddingBottom: 4, borderBottom: '1px solid oklch(26% 0.02 240)' },
  sweepSummary: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  sweepChip: { display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px', borderRadius: 5, fontSize: 12, fontWeight: 500, cursor: 'pointer', border: '1px solid', userSelect: 'none' },
  sweepChipOn: { background: 'oklch(68% 0.19 145 / 0.12)', color: 'oklch(78% 0.17 145)', borderColor: 'oklch(68% 0.19 145 / 0.35)' },
  sweepChipOff: { background: 'oklch(20% 0.018 240)', color: 'oklch(44% 0.012 240)', borderColor: 'oklch(26% 0.02 240)' },
  comboCount: { marginLeft: 'auto', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(78% 0.18 75)', fontWeight: 600 },
  paramsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 },
  paramBlock: { background: 'oklch(16% 0.016 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 6, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 7 },
  paramLabel: { fontSize: 12, fontWeight: 600, color: 'oklch(94% 0.005 240)' },
  paramDefault: { fontSize: 10, color: 'oklch(44% 0.012 240)', fontWeight: 400, marginLeft: 6 },
  rangeRow: { display: 'flex', gap: 6 },
  rangeField: { flex: 1, display: 'flex', flexDirection: 'column', gap: 3 },
  rangeLabel: { fontSize: 10, color: 'oklch(44% 0.012 240)' },
  rangeInput: { background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 4, padding: '5px 7px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'oklch(94% 0.005 240)', outline: 'none', width: '100%' },
  catRow: { display: 'flex', gap: 10, flexWrap: 'wrap' },
  catOption: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'oklch(94% 0.005 240)', cursor: 'pointer' },
  radioGroup: { display: 'flex', gap: 2, background: 'oklch(20% 0.018 240)', border: '1px solid oklch(26% 0.02 240)', borderRadius: 5, padding: 3 },
  radioOpt: { flex: 1, padding: '4px 8px', borderRadius: 4, fontSize: 11, fontWeight: 500, color: 'oklch(65% 0.012 240)', textAlign: 'center', cursor: 'pointer' },
  radioOptActive: { background: 'oklch(78% 0.18 75)', color: 'oklch(10% 0.01 240)', fontWeight: 600 },
  splitRow: { paddingTop: 4 },
  splitBar: { display: 'flex', borderRadius: 4, overflow: 'hidden', height: 26 },
  splitIS: { flex: 7, background: 'oklch(65% 0.155 250 / 0.25)', color: 'oklch(72% 0.14 250)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 600 },
  splitOOS: { flex: 3, background: 'oklch(78% 0.18 75 / 0.2)', color: 'oklch(78% 0.18 75)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 600 },
  runBar: { display: 'flex', alignItems: 'center', gap: 14, paddingTop: 4, borderTop: '1px solid oklch(26% 0.02 240)', marginTop: 4 },
  btnPrimary: { padding: '9px 20px', background: 'oklch(78% 0.18 75)', color: 'oklch(10% 0.01 240)', border: 'none', borderRadius: 5, fontSize: 14, fontWeight: 700, fontFamily: "'Inter', sans-serif", cursor: 'pointer', boxShadow: '0 0 14px oklch(78% 0.18 75 / 0.3)', letterSpacing: '0.01em' },
  btnSecondary: { padding: '7px 14px', background: 'oklch(20% 0.018 240)', color: 'oklch(94% 0.005 240)', border: '1px solid oklch(32% 0.022 240)', borderRadius: 5, fontSize: 12, fontWeight: 500, fontFamily: "'Inter', sans-serif", cursor: 'pointer', flexShrink: 0 },
  btnDanger: { padding: '9px 20px', background: 'oklch(60% 0.22 25 / 0.15)', color: 'oklch(70% 0.2 25)', border: '1px solid oklch(60% 0.22 25 / 0.4)', borderRadius: 5, fontSize: 14, fontWeight: 700, fontFamily: "'Inter', sans-serif", cursor: 'pointer' },
  successMsg: { fontSize: 13, color: 'oklch(78% 0.17 145)' },
  progressWrap: { flex: 1, display: 'flex', flexDirection: 'column', gap: 4 },
  progressLabel: { fontSize: 11, color: 'oklch(65% 0.012 240)', fontFamily: "'JetBrains Mono', monospace" },
  progressBar: { height: 4, background: 'oklch(20% 0.018 240)', borderRadius: 2, overflow: 'hidden' },
  progressFill: { height: '100%', background: 'oklch(78% 0.18 75)', borderRadius: 2, transition: 'width 300ms ease' },
};

Object.assign(window, { ConfigureRun, crStyles, MOCK_PARAMS });
