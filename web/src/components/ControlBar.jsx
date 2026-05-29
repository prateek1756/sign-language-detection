/**
 * ControlBar.jsx — Start / Stop / Mode toggle / Dialect selector.
 */
export default function ControlBar({
  isActive,
  wsStatus,
  mode,
  dialect,
  onStart,
  onStop,
  onModeChange,
  onDialectChange,
}) {

  return (
    <nav className="control-bar" role="toolbar" aria-label="Camera and mode controls">

      {/* Camera Start/Stop */}
      {!isActive ? (
        <button
          id="btn-start"
          className="btn btn-primary"
          onClick={onStart}
          aria-label="Start camera and begin detection"
        >
          ▶ Start
        </button>
      ) : (
        <button
          id="btn-stop"
          className="btn btn-danger"
          onClick={onStop}
          aria-label="Stop camera"
        >
          ⏹ Stop
        </button>
      )}

      {/* Divider */}
      <div style={{ width: 1, height: 24, background: 'var(--clr-card-border)' }} aria-hidden="true" />

      {/* Mode: Letter / Word */}
      <div role="group" aria-label="Prediction mode" style={{ display: 'flex', gap: '6px' }}>
        <button
          id="btn-mode-letter"
          className={`btn btn-mode ${mode === 'letter' ? 'active' : ''}`}
          onClick={() => onModeChange('letter')}
          aria-pressed={mode === 'letter'}
          aria-label="Letter mode — predict individual signs"
        >
          ✋ Letter
        </button>
        <button
          id="btn-mode-word"
          className={`btn btn-mode ${mode === 'word' ? 'active' : ''}`}
          onClick={() => onModeChange('word')}
          aria-pressed={mode === 'word'}
          aria-label="Word mode — predict 30-frame gesture sequences"
        >
          🌊 Word
        </button>
      </div>

      {/* Divider */}
      <div style={{ width: 1, height: 24, background: 'var(--clr-card-border)' }} aria-hidden="true" />

      {/* Dialect selector */}
      <label htmlFor="dialect-select" style={{ fontSize: '0.8rem', color: 'var(--clr-text-muted)' }}>
        Language
      </label>
      <select
        id="dialect-select"
        className="select"
        value={dialect}
        onChange={e => onDialectChange(e.target.value)}
        aria-label="Select sign language dialect"
      >
        <option value="ASL">ASL</option>
        <option value="ISL" disabled>ISL (soon)</option>
        <option value="BSL" disabled>BSL (soon)</option>
      </select>

      {/* Connection status */}
      <div
        className={`status-badge ${wsStatus}`}
        role="status"
        aria-live="polite"
        aria-label={`Backend connection: ${wsStatus}`}
      >
        <span className="status-badge__dot" aria-hidden="true" />
        {wsStatus === 'connected'   && 'Live'}
        {wsStatus === 'connecting'  && 'Connecting…'}
        {wsStatus === 'disconnected'&& 'Offline'}
        {wsStatus === 'error'       && 'Error'}
      </div>
    </nav>
  )
}
