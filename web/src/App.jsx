/**
 * App.jsx — Root application shell (React 19).
 *
 * State orchestration:
 *  - useCamera   → webcam stream + frame capture
 *  - useWebSocket→ WS connection + frame streaming
 *  - Prediction result → drives PredictionPanel + SentenceBuilder
 *  - Frame capture interval: 150ms (≈6 fps — sufficient for ASL letter detection)
 *
 * @kaizen improvements applied:
 *  - Frame capture only when WS is connected (no wasted encodes)
 *  - useCallback on all handler functions (stable refs)
 *  - Cleanup via useEffect return (no memory leaks)
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { useCamera }    from './hooks/useCamera.js'
import { useWebSocket } from './hooks/useWebSocket.js'
import CameraFeed       from './components/CameraFeed.jsx'
import PredictionPanel  from './components/PredictionPanel.jsx'
import SentenceBuilder  from './components/SentenceBuilder.jsx'
import ControlBar       from './components/ControlBar.jsx'
import SignSenseWidget  from './components/SignSenseWidget.jsx'
import SessionAnalytics from './components/SessionAnalytics.jsx'

const FRAME_INTERVAL_MS = 150  // 6–7 fps — good balance for ASL detection latency

export default function App() {
  const [mode,    setMode]    = useState('letter')  // 'letter' | 'word'
  const [dialect, setDialect] = useState('ASL')
  const [result,  setResult]  = useState(null)
  const [viewMode, setViewMode] = useState('app')     // 'app' | 'widget'
  const [widgetMode, setWidgetMode] = useState('letter') // 'letter' | 'word'

  const frameTimerRef = useRef(null)

  const {
    videoRef, isActive, isLoading, error: cameraError,
    startCamera, stopCamera, captureFrame,
  } = useCamera()

  const {
    status: wsStatus, lastResult,
    connect, disconnect, sendFrame,
  } = useWebSocket()

  const lastLoggedLetterRef = useRef(null)
  const [history, setHistory] = useState([])

  // Sync lastResult → local result state + track session history
  useEffect(() => {
    if (lastResult) {
      setResult(lastResult)
      
      const letter = lastResult.letter ?? lastResult.word
      if (letter && letter !== 'nothing' && letter !== lastLoggedLetterRef.current && !lastResult.buffering) {
        lastLoggedLetterRef.current = letter
        const now = new Date()
        const timeStr = now.toTimeString().split(' ')[0]
        setHistory(prev => [
          ...prev,
          {
            letter,
            confidence: lastResult.confidence ?? 1.0,
            time: timeStr
          }
        ])
      } else if (!letter || letter === 'nothing') {
        lastLoggedLetterRef.current = null
      }
    }
  }, [lastResult])

  // Start frame capture interval when both camera and WS are ready
  useEffect(() => {
    if (isActive && wsStatus === 'connected' && viewMode === 'app') {
      frameTimerRef.current = setInterval(() => {
        const frame = captureFrame()
        if (frame) sendFrame(frame, mode)
      }, FRAME_INTERVAL_MS)
    } else {
      clearInterval(frameTimerRef.current)
      frameTimerRef.current = null
    }
    return () => clearInterval(frameTimerRef.current)
  }, [isActive, wsStatus, mode, captureFrame, sendFrame, viewMode])

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleStart = useCallback(async () => {
    await startCamera()
    connect()
  }, [startCamera, connect])

  const handleStop = useCallback(() => {
    clearInterval(frameTimerRef.current)
    stopCamera()
    disconnect()
    setResult(null)
  }, [stopCamera, disconnect])

  const handleModeChange = useCallback((newMode) => {
    setMode(newMode)
    setResult(null)  // clear stale result when switching modes
  }, [])

  const handleClear = useCallback(() => {
    setResult(null)
  }, [])

  // ── Landmarks from latest result ──────────────────────────────────────────
  const landmarks = result?.landmarks ?? null

  return (
    <div className="app" id="app-root">

      {/* ── Header ── */}
      <header className="header" role="banner">
        <div className="header__brand">
          <div className="header__logo" aria-hidden="true">🤟</div>
          <div>
            <div className="header__title">SignSense AI</div>
            <div className="header__subtitle">Real-Time Sign Language Detection</div>
          </div>
        </div>
        <div className="header__controls" style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setViewMode('app')}
              className={`btn btn-mode ${viewMode === 'app' ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            >
              Full Platform
            </button>
            <button
              onClick={() => {
                handleStop()
                setViewMode('widget')
              }}
              className={`btn btn-mode ${viewMode === 'widget' ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            >
              Embed Widget Demo
            </button>
          </div>
          <span
            className={`status-badge ${wsStatus}`}
            role="status"
            aria-live="polite"
            aria-label={`Connection status: ${wsStatus}`}
            style={{ display: viewMode === 'app' ? 'inline-flex' : 'none' }}
          >
            <span className="status-badge__dot" aria-hidden="true" />
            {wsStatus === 'connected'    && 'API Connected'}
            {wsStatus === 'connecting'   && 'Connecting…'}
            {wsStatus === 'disconnected' && 'API Offline'}
            {wsStatus === 'error'        && 'Connection Error'}
          </span>
        </div>
      </header>

      {/* ── Conditional Rendering based on viewMode ── */}
      {viewMode === 'app' ? (
        <>
          {/* ── Main Grid ── */}
          <main className="main-grid" role="main" id="main-content">

            {/* Left column: Camera */}
            <section aria-label="Camera feed">
              <div className="card">
                <div className="card__header">
                  <span className="card__title">📷 Camera</span>
                  {isActive && (
                    <span
                      style={{
                        display: 'inline-flex', alignItems: 'center', gap: '6px',
                        fontSize: '0.72rem', color: 'var(--clr-emerald)', fontWeight: 600,
                      }}
                    >
                      <span style={{
                        width: 7, height: 7, borderRadius: '50%',
                        background: 'var(--clr-emerald)',
                        animation: 'dot-pulse 2s ease-in-out infinite',
                      }} />
                      LIVE
                    </span>
                  )}
                </div>
                <div className="card__body" style={{ padding: '12px' }}>
                  <CameraFeed
                    videoRef={videoRef}
                    isActive={isActive}
                    isLoading={isLoading}
                    landmarks={landmarks}
                  />
                  {cameraError && (
                    <div className="error-msg" role="alert" style={{ marginTop: '12px' }}>
                      ⚠ {cameraError}
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Right column: Prediction */}
            <aside aria-label="Predictions" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="card" style={{ flex: 1 }}>
                <div className="card__header">
                  <span className="card__title">🧠 Prediction</span>
                  <span className={`mode-badge ${mode}`}>
                    {mode === 'word' ? '🌊' : '✋'} {mode}
                  </span>
                </div>
                <div className="card__body">
                  <PredictionPanel result={result} mode={mode} />
                </div>
              </div>

              {/* Tips card */}
              <div className="card">
                <div className="card__header">
                  <span className="card__title">💡 Tips</span>
                </div>
                <div className="card__body" style={{ padding: '12px 16px' }}>
                  <ul style={{
                    fontSize: '0.78rem',
                    color: 'var(--clr-text-muted)',
                    paddingLeft: '1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                  }}>
                    <li>Ensure your hand is fully visible and well-lit</li>
                    <li>Hold each sign steady for ~1.5s to commit it</li>
                    <li>Use <strong>Word mode</strong> for dynamic gestures</li>
                    <li>Click <strong>🔊 Speak</strong> to hear your sentence</li>
                  </ul>
                </div>
              </div>
            </aside>

            {/* ── Session Analytics Panel ── */}
            <SessionAnalytics
              history={history}
              onClear={() => setHistory([])}
            />
          </main>

          {/* ── Sentence Builder ── */}
          <SentenceBuilder
            currentLetter={result?.letter ?? result?.word ?? null}
            onClear={handleClear}
          />

          {/* ── Control Bar ── */}
          <ControlBar
            isActive={isActive}
            wsStatus={wsStatus}
            mode={mode}
            dialect={dialect}
            onStart={handleStart}
            onStop={handleStop}
            onModeChange={handleModeChange}
            onDialectChange={setDialect}
          />
        </>
      ) : (
        <div style={{ display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px 20px', gap: '20px' }}>
          <div style={{ textAlign: 'center', maxWidth: '560px' }}>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, marginBottom: '8px', background: 'linear-gradient(135deg, #fff 0%, var(--clr-accent) 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Embeddable Client-Side SaaS Widget
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--clr-text-muted)', lineHeight: 1.6 }}>
              This neomorphic widget runs both MediaPipe landmark tracking and neural net classification 
              <strong> 100% locally in the browser</strong> at 0ms network transit and $0 server cost.
            </p>
          </div>
          
          <div style={{ display: 'flex', gap: '10px', marginBottom: '8px' }}>
            <button
              onClick={() => setWidgetMode('letter')}
              className={`btn btn-mode ${widgetMode === 'letter' ? 'active' : ''}`}
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
            >
              ✋ Letter Mode (MLP)
            </button>
            <button
              onClick={() => setWidgetMode('word')}
              className={`btn btn-mode ${widgetMode === 'word' ? 'active' : ''}`}
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
            >
              🌊 Word Mode (BiLSTM)
            </button>
          </div>

          <SignSenseWidget
            mode={widgetMode}
            width={640}
            height={480}
            onPrediction={(label, conf) => {
              console.log(`Widget predicted gesture: ${label} (Confidence: ${conf})`)
            }}
          />
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="footer" role="contentinfo">
        SignSense AI · ASL Detection · B2B SaaS Widget Demo · React 19 + ONNX Runtime Web
      </footer>
    </div>
  )
}
