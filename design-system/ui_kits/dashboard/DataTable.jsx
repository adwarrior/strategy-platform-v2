// DataTable — sortable results table

function DataTable({ columns, rows, selectedRow, onRowClick, maxHeight = 300 }) {
  const [sortCol, setSortCol] = React.useState(null);
  const [sortAsc, setSortAsc] = React.useState(false);

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(false); }
  };

  const sorted = React.useMemo(() => {
    if (!sortCol) return rows;
    return [...rows].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol];
      if (av == null) return 1;
      if (bv == null) return -1;
      return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });
  }, [rows, sortCol, sortAsc]);

  const formatCell = (val, col) => {
    if (val == null) return '—';
    if (col.type === 'dollar') return (val >= 0 ? '+' : '') + '$' + Math.abs(val).toLocaleString(undefined, {maximumFractionDigits:0});
    if (col.type === 'pct') return (val * 100).toFixed(1) + '%';
    if (col.type === 'float') return val.toFixed(2);
    if (col.type === 'int') return val.toLocaleString();
    return String(val);
  };

  const cellColor = (val, col) => {
    if (col.colored && typeof val === 'number') {
      if (val > 0) return 'oklch(78% 0.17 145)';
      if (val < 0) return 'oklch(70% 0.2 25)';
    }
    if (col.type === 'code') return 'oklch(72% 0.145 195)';
    return 'oklch(94% 0.005 240)';
  };

  return (
    <div style={dataTableStyles.wrap}>
      <table style={dataTableStyles.table}>
        <thead>
          <tr style={dataTableStyles.headRow}>
            {columns.map(col => (
              <th
                key={col.key}
                style={{...dataTableStyles.th, ...(col.numeric ? dataTableStyles.thNum : {}), cursor:'pointer'}}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortCol === col.key && <span style={{marginLeft:4, opacity:0.7}}>{sortAsc ? '↑' : '↓'}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={i}
              style={{
                ...dataTableStyles.row,
                ...(selectedRow === i ? dataTableStyles.selectedRow : {}),
              }}
              onClick={() => onRowClick && onRowClick(i, row)}
            >
              {columns.map(col => (
                <td
                  key={col.key}
                  style={{
                    ...dataTableStyles.td,
                    ...(col.numeric ? dataTableStyles.tdNum : {}),
                    ...(col.type === 'code' ? dataTableStyles.tdCode : {}),
                    color: cellColor(row[col.key], col),
                  }}
                >
                  {col.type === 'badge' ? (
                    <span style={{...dataTableStyles.badge, ...getBadgeStyle(row[col.key])}}>{row[col.key]}</span>
                  ) : formatCell(row[col.key], col)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function getBadgeStyle(val) {
  if (val === 'Shortlisted') return { background:'oklch(68% 0.19 145 / 0.15)', color:'oklch(78% 0.17 145)', border:'1px solid oklch(68% 0.19 145 / 0.35)' };
  if (val === 'Rejected')    return { background:'oklch(60% 0.22 25 / 0.15)',  color:'oklch(70% 0.2 25)',   border:'1px solid oklch(60% 0.22 25 / 0.35)' };
  return { background:'oklch(20% 0.018 240)', color:'oklch(65% 0.012 240)', border:'1px solid oklch(32% 0.022 240)' };
}

const dataTableStyles = {
  wrap: {
    border: '1px solid oklch(26% 0.02 240)',
    borderRadius: 7,
    overflow: 'hidden',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  headRow: {
    background: 'oklch(20% 0.018 240)',
  },
  th: {
    padding: '8px 12px',
    textAlign: 'left',
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    color: 'oklch(65% 0.012 240)',
    borderBottom: '1px solid oklch(26% 0.02 240)',
    whiteSpace: 'nowrap',
    fontFamily: "'Inter', sans-serif",
    userSelect: 'none',
  },
  thNum: {
    textAlign: 'right',
  },
  row: {
    borderBottom: '1px solid oklch(26% 0.02 240 / 0.6)',
    cursor: 'pointer',
    transition: 'background 80ms',
  },
  selectedRow: {
    background: 'oklch(78% 0.18 75 / 0.06)',
    borderLeft: '2px solid oklch(78% 0.18 75)',
  },
  td: {
    padding: '7px 12px',
    fontSize: 12,
    fontFamily: "'Inter', sans-serif",
    color: 'oklch(94% 0.005 240)',
  },
  tdNum: {
    textAlign: 'right',
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 12,
  },
  tdCode: {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 11,
    color: 'oklch(72% 0.145 195)',
  },
  badge: {
    display: 'inline-block',
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 7px',
    borderRadius: 9999,
    fontFamily: "'JetBrains Mono', monospace",
  },
};

Object.assign(window, { DataTable, dataTableStyles, getBadgeStyle });
