/**
 * SessionAnalytics.jsx — Live Session Statistics and Gesture History log.
 *
 * Visualizations:
 *  - Scrollable list of recent commitments (time + prediction + confidence)
 *  - Horizontal bar chart of gesture frequencies
 *  - High-precision performance indicators (total commitments, top gesture)
 */
import { useMemo } from 'react'

export default function SessionAnalytics({ history, onClear }) {
  // Compute sign frequencies
  const frequencies = useMemo(() => {
    const counts = {}
    for (const item of history) {
      if (item.letter) {
        counts[item.letter] = (counts[item.letter] || 0) + 1
      }
    }
    return Object.entries(counts)
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count)
  }, [history])

  // Total signs predicted in the active session
  const totalPredictions = history.length

  // Identify the most frequent gesture
  const topGesture = useMemo(() => {
    if (frequencies.length === 0) return 'None'
    const best = frequencies[0]
    const percent = Math.round((best.count / totalPredictions) * 100)
    return `${best.label} (${percent}%)`
  }, [frequencies, totalPredictions])

  // Get max count for scaling bar chart widths
  const maxCount = useMemo(() => {
    if (frequencies.length === 0) return 1
    return Math.max(...frequencies.map(f => f.count))
  }, [frequencies])

  return (
    <div className="card analytics-card" id="session-analytics" style={{ gridColumn: 'span 2' }}>
      <div className="card__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="card__title">📊 Live Session Analytics</span>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="btn btn-mode"
            style={{
              padding: '4px 10px',
              fontSize: '0.72rem',
              background: 'rgba(239, 68, 68, 0.15)',
              color: '#ef4444',
              border: '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: 6,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            Clear Stats
          </button>
        )}
      </div>

      <div className="card__body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', padding: '16px 20px' }}>
        
        {/* Left Column: History Log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--clr-text-muted)', borderBottom: '1px solid var(--clr-card-border)', paddingBottom: '6px' }}>
            📜 Recent Activity Log
          </div>
          {history.length === 0 ? (
            <div style={{ display: 'flex', flex: 1, alignItems: 'center', justifyContent: 'center', minHeight: '160px', fontSize: '0.82rem', color: 'var(--clr-text-muted)', textAlign: 'center', border: '1px dashed var(--clr-card-border)', borderRadius: 8 }}>
              No gestures detected in this session yet.<br />Click Start and gesture to begin.
            </div>
          ) : (
            <div style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', paddingRight: '4px' }}>
              {history.slice().reverse().map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 12px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid var(--clr-card-border)',
                    borderRadius: 6,
                    fontSize: '0.78rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--clr-text-muted)', fontSize: '0.72rem' }}>{item.time}</span>
                    <span style={{ fontWeight: 700, color: 'var(--clr-accent)', background: 'rgba(124, 92, 252, 0.1)', padding: '2px 6px', borderRadius: 4 }}>
                      {item.letter}
                    </span>
                  </div>
                  <span style={{ color: 'var(--clr-emerald)', fontWeight: 600 }}>
                    {Math.round(item.confidence * 100)}% conf
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column: Frequencies & Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Metadata Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--clr-card-border)', padding: '10px', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--clr-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Signs</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--clr-accent)', marginTop: '4px' }}>{totalPredictions}</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--clr-card-border)', padding: '10px', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--clr-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Top Gesture</div>
              <div style={{ fontSize: '0.98rem', fontWeight: 800, color: 'var(--clr-emerald)', marginTop: '8px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{topGesture}</div>
            </div>
          </div>

          {/* Horizontal Bar Chart */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--clr-text-muted)', borderBottom: '1px solid var(--clr-card-border)', paddingBottom: '6px' }}>
              📊 Gesture Distribution
            </div>
            {frequencies.length === 0 ? (
              <div style={{ display: 'flex', flex: 1, alignItems: 'center', justifyContent: 'center', fontSize: '0.82rem', color: 'var(--clr-text-muted)', minHeight: '110px' }}>
                Waiting for data...
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '110px', overflowY: 'auto', paddingRight: '4px' }}>
                {frequencies.slice(0, 4).map((f) => {
                  const barWidth = Math.max(10, Math.round((f.count / maxCount) * 100))
                  return (
                    <div key={f.label} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.76rem' }}>
                      <span style={{ width: '16px', fontWeight: 700, textAlign: 'center', color: 'var(--clr-accent)' }}>{f.label}</span>
                      <div style={{ flex: 1, height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: 99, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.02)' }}>
                        <div
                          style={{
                            width: `${barWidth}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, var(--clr-accent) 0%, var(--clr-emerald) 100%)',
                            borderRadius: 99,
                            transition: 'width 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
                          }}
                        />
                      </div>
                      <span style={{ width: '28px', color: 'var(--clr-text-muted)', textAlign: 'right', fontSize: '0.7rem' }}>
                        x{f.count}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

        </div>

      </div>
    </div>
  )
}
