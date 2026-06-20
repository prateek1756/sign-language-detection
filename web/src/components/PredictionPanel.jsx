/**
 * PredictionPanel.jsx — Displays current sign prediction.
 *
 * Shows:
 *  - Large predicted letter with pop animation
 *  - Confidence bar (colour-coded: emerald/violet/amber)
 *  - Top-3 alternatives with mini bars
 *  - Mode badge (LETTER / WORD)
 *  - Latency tag
 *  - "No hand detected" warning
 *  - Word buffering progress bar
 */
import { useRef, useEffect, useState } from 'react'

function confidenceClass(conf) {
  if (conf >= 0.85) return 'high'
  if (conf >= 0.70) return 'medium'
  return 'low'
}

export default function PredictionPanel({ result, mode }) {
  const prevLetterRef = useRef(null)
  const [popKey, setPopKey] = useState(0)

  const letter     = result?.letter ?? null
  const confidence = result?.confidence ?? 0
  const top3       = result?.top3 ?? []
  const latency    = result?.latency_ms ?? null
  const noHand     = result?.hand_detected === false
  const buffering  = result?.buffering === true
  const below      = result?.below_threshold === true

  // Trigger pop animation when letter changes
  useEffect(() => {
    if (letter && letter !== prevLetterRef.current) {
      prevLetterRef.current = letter
      setPopKey(k => k + 1)
    }
  }, [letter])

  const displayLetter = letter === 'nothing' ? null : letter === 'space' ? '⎵' : letter === 'del' ? '⌫' : letter ?? (below ? '?' : null)
  const pct = Math.round(confidence * 100)
  const confClass = confidenceClass(confidence)

  return (
    <div className="prediction-panel" id="prediction-panel" role="region" aria-label="Sign prediction">

      {/* Letter display */}
      <div className={`letter-display ${displayLetter ? 'active' : ''}`}>
        <div className="letter-display__glow" aria-hidden="true" />
        {displayLetter ? (
          <span
            key={popKey}
            className="letter-display__char pop"
            aria-live="polite"
            aria-atomic="true"
          >
            {displayLetter}
          </span>
        ) : (
          <span className="letter-display__empty" aria-hidden="true">—</span>
        )}
        <span className="letter-display__label">
          {mode === 'word' ? 'Word' : 'Letter'}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="confidence-section" aria-label={`Confidence: ${pct}%`}>
        <div className="confidence-row">
          <span>Confidence</span>
          <span
            className="confidence-value"
            style={{ color: confClass === 'high' ? 'var(--clr-emerald)' : confClass === 'medium' ? 'var(--clr-accent)' : 'var(--clr-amber)' }}
          >
            {pct}%
          </span>
        </div>
        <div className="confidence-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div
            className={`confidence-bar__fill ${confClass}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Warnings */}
      {noHand && (
        <div className="no-hand-banner" role="alert" aria-live="polite">
          ✋ No hand detected — position your hand in the camera frame
        </div>
      )}
      {below && !noHand && confidence > 0 && (
        <div className="no-hand-banner" role="status">
          ⚠ Low confidence ({pct}%) — try a clearer gesture
        </div>
      )}

      {/* Word buffering bar */}
      {buffering && (
        <div className="buffering-bar" role="status" aria-live="polite">
          <span>Buffering</span>
          <div className="buffering-bar__track">
            <div
              className="buffering-bar__fill"
              style={{ width: `${(result.frames_buffered / result.frames_needed) * 100}%` }}
            />
          </div>
          <span>{result.frames_buffered}/{result.frames_needed}</span>
        </div>
      )}

      {/* Mode badge + latency */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className={`mode-badge ${mode}`} aria-label={`Mode: ${mode}`}>
          <span aria-hidden="true">{mode === 'word' ? '🌊' : '✋'}</span>
          {mode} mode
        </span>
        {latency !== null && (
          <span className="latency-tag" title="Inference latency">
            ⚡ {latency}ms
          </span>
        )}
      </div>

      {/* Top-3 alternatives */}
      {top3.length > 0 && (
        <div className="top3" aria-label="Top 3 predictions">
          <div className="card__title" style={{ marginBottom: '4px' }}>Top alternatives</div>
          {top3.map((item) => (
            <div key={item.label} className="top3__item" role="listitem">
              <span className="top3__label">{item.label}</span>
              <div className="top3__bar" aria-hidden="true">
                <div
                  className="top3__bar-fill"
                  style={{ width: `${Math.round(item.confidence * 100)}%` }}
                />
              </div>
              <span className="top3__conf">{Math.round(item.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
