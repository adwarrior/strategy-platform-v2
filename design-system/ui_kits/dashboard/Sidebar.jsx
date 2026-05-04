// Sidebar Component
// strategy selector, bar type, symbol, grid info

function Sidebar({ selectedStrategy, onStrategyChange, strategies, symbol, gridCombos, runTss }) {
  const [barCategory, setBarCategory] = React.useState('Minute bars');
  const [minuteInc, setMinuteInc] = React.useState(5);

  return (
    <div style={sidebarStyles.sidebar}>
      <div style={sidebarStyles.brand}>
        <div style={sidebarStyles.brandDot}></div>
        <span style={sidebarStyles.brandName}>Strategy Platform</span>
      </div>
      <div style={sidebarStyles.divider}></div>

      <div style={sidebarStyles.field}>
        <div style={sidebarStyles.fieldLabel}>Strategy</div>
        <select
          style={sidebarStyles.select}
          value={selectedStrategy}
          onChange={e => onStrategyChange(e.target.value)}
        >
          {strategies.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <div style={sidebarStyles.field}>
        <div style={sidebarStyles.fieldLabel}>Bar Type</div>
        <select
          style={sidebarStyles.select}
          value={barCategory}
          onChange={e => setBarCategory(e.target.value)}
        >
          <option>Minute bars</option>
          <option>Tick bars</option>
        </select>
      </div>

      {barCategory === 'Minute bars' && (
        <div style={sidebarStyles.field}>
          <div style={sidebarStyles.fieldLabel}>Minutes per bar</div>
          <div style={sidebarStyles.stepRow}>
            {[1,2,3,4,5].map(n => (
              <div
                key={n}
                style={{...sidebarStyles.stepBtn, ...(minuteInc === n ? sidebarStyles.stepBtnActive : {})}}
                onClick={() => setMinuteInc(n)}
              >{n}M</div>
            ))}
          </div>
        </div>
      )}

      <div style={sidebarStyles.divider}></div>

      <div style={sidebarStyles.infoRow}>
        <div style={sidebarStyles.infoDot}></div>
        <span style={sidebarStyles.infoText}>{symbol} · {minuteInc}M bars</span>
      </div>
      <div style={sidebarStyles.statRow}>
        <span style={sidebarStyles.statLabel}>Run history</span>
        <span style={sidebarStyles.statVal}>{runTss.length} runs</span>
      </div>
      <div style={sidebarStyles.statRow}>
        <span style={sidebarStyles.statLabel}>Grid size</span>
        <span style={sidebarStyles.statVal}>{gridCombos.toLocaleString()} combos</span>
      </div>

      <div style={{flex:1}}></div>

      <div style={sidebarStyles.reportPath}>
        <span style={sidebarStyles.reportPathText}>reports/</span>
      </div>
    </div>
  );
}

const sidebarStyles = {
  sidebar: {
    width: 230,
    minWidth: 230,
    background: 'oklch(14% 0.014 240)',
    borderRight: '1px solid oklch(26% 0.02 240)',
    padding: '16px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    overflowY: 'auto',
    height: '100%',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  brandDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: 'oklch(78% 0.18 75)',
    flexShrink: 0,
    boxShadow: '0 0 8px oklch(78% 0.18 75 / 0.5)',
  },
  brandName: {
    fontFamily: "'Space Grotesk', sans-serif",
    fontSize: 15,
    fontWeight: 700,
    color: 'oklch(94% 0.005 240)',
    letterSpacing: '-0.01em',
  },
  divider: {
    height: 1,
    background: 'oklch(26% 0.02 240)',
    margin: '2px 0',
  },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
  },
  fieldLabel: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: '0.09em',
    textTransform: 'uppercase',
    color: 'oklch(44% 0.012 240)',
  },
  select: {
    appearance: 'none',
    WebkitAppearance: 'none',
    background: 'oklch(20% 0.018 240)',
    border: '1px solid oklch(26% 0.02 240)',
    borderRadius: 4,
    padding: '6px 22px 6px 9px',
    fontSize: 12,
    fontFamily: "'Inter', sans-serif",
    color: 'oklch(94% 0.005 240)',
    width: '100%',
    cursor: 'pointer',
  },
  stepRow: {
    display: 'flex',
    gap: 3,
  },
  stepBtn: {
    flex: 1,
    padding: '5px 0',
    textAlign: 'center',
    fontSize: 11,
    fontWeight: 600,
    background: 'oklch(20% 0.018 240)',
    border: '1px solid oklch(26% 0.02 240)',
    borderRadius: 4,
    color: 'oklch(65% 0.012 240)',
    cursor: 'pointer',
    transition: 'all 100ms',
    fontFamily: "'JetBrains Mono', monospace",
  },
  stepBtnActive: {
    background: 'oklch(78% 0.18 75 / 0.15)',
    borderColor: 'oklch(78% 0.18 75 / 0.6)',
    color: 'oklch(78% 0.18 75)',
  },
  infoRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  infoDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'oklch(68% 0.19 145)',
    flexShrink: 0,
  },
  infoText: {
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
    color: 'oklch(94% 0.005 240)',
    fontWeight: 500,
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 11,
    color: 'oklch(44% 0.012 240)',
  },
  statVal: {
    fontSize: 11,
    fontFamily: "'JetBrains Mono', monospace",
    color: 'oklch(65% 0.012 240)',
  },
  reportPath: {
    padding: '6px 0',
  },
  reportPathText: {
    fontSize: 10,
    fontFamily: "'JetBrains Mono', monospace",
    color: 'oklch(38% 0.02 240)',
  },
};

Object.assign(window, { Sidebar, sidebarStyles });
