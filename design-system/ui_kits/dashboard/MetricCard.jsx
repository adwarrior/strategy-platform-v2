// MetricCard — KPI card component used throughout the dashboard

function MetricCard({ label, value, sub, trend, size = 'default', accent }) {
  // trend: 'pos' | 'neg' | 'neu'
  // accent: 'amber' | 'teal' | 'green' | 'red' | null
  const accentColors = {
    amber: 'oklch(78% 0.18 75)',
    teal:  'oklch(72% 0.145 195)',
    green: 'oklch(78% 0.17 145)',
    red:   'oklch(70% 0.2 25)',
  };
  const trendColors = {
    pos: 'oklch(78% 0.17 145)',
    neg: 'oklch(70% 0.2 25)',
    neu: 'oklch(94% 0.005 240)',
  };

  const valueColor = accent ? accentColors[accent] : (trend ? trendColors[trend] : trendColors.neu);
  const valueSizes = { default: 24, large: 32, mini: 18 };
  const valueSize = valueSizes[size] || 24;

  return (
    <div style={metricCardStyles.card}>
      <div style={metricCardStyles.label}>{label}</div>
      <div style={{...metricCardStyles.value, fontSize: valueSize, color: valueColor}}>
        {value}
      </div>
      {sub && <div style={{...metricCardStyles.sub, color: trend ? trendColors[trend] : 'oklch(65% 0.012 240)'}}>{sub}</div>}
    </div>
  );
}

const metricCardStyles = {
  card: {
    background: 'oklch(16% 0.016 240)',
    border: '1px solid oklch(26% 0.02 240)',
    borderRadius: 7,
    padding: '10px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
  },
  label: {
    fontSize: 11,
    fontWeight: 500,
    color: 'oklch(65% 0.012 240)',
    letterSpacing: '0.03em',
    textTransform: 'uppercase',
    fontFamily: "'Inter', sans-serif",
  },
  value: {
    fontFamily: "'JetBrains Mono', monospace",
    fontWeight: 600,
    lineHeight: 1.1,
    color: 'oklch(94% 0.005 240)',
  },
  sub: {
    fontSize: 11,
    fontFamily: "'JetBrains Mono', monospace",
    color: 'oklch(65% 0.012 240)',
  },
};

Object.assign(window, { MetricCard, metricCardStyles });
