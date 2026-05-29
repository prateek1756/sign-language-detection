/**
 * SentenceBuilder.jsx — Accumulates letters/words into a sentence.
 *
 * Features:
 *  - Auto-commits stable letter after 1.5s hold (kaizen: reduces false positives)
 *  - SPACE / BACKSPACE / CLEAR actions
 *  - TTS via SpeechSynthesis API (browser-native, no library needed)
 *  - Animated character entry
 *  - Typewriter cursor
 *
 * B-SB1 FIX: chars stored as [{id, char}] array instead of a plain string.
 * This gives each character a stable unique key, preventing React reconciliation
 * bugs (wrong character animating) when backspace removes from the middle/end.
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const AUTO_COMMIT_MS = 1_500  // hold same letter for this long to auto-commit

export default function SentenceBuilder({ currentLetter, onClear }) {
  // B-SB1 FIX: [{id: number, char: string}] instead of plain string
  const [chars,        setChars]        = useState([])
  const [speaking,     setSpeaking]     = useState(false)
  const holdTimer      = useRef(null)
  const lastLetterRef  = useRef(null)
  const holdStartRef   = useRef(null)
  const nextIdRef      = useRef(0)  // monotonically increasing stable key

  // Derived plain string for TTS and aria-label
  const sentence = chars.map(c => c.char).join('')

  // Auto-commit: if the same letter is held for AUTO_COMMIT_MS, append it
  useEffect(() => {
    if (!currentLetter || currentLetter === lastLetterRef.current) return
    lastLetterRef.current = currentLetter

    clearTimeout(holdTimer.current)
    holdStartRef.current = Date.now()

    holdTimer.current = setTimeout(() => {
      const id = nextIdRef.current++
      setChars(prev => [...prev, { id, char: currentLetter }])
    }, AUTO_COMMIT_MS)

    return () => clearTimeout(holdTimer.current)
  }, [currentLetter])

  const handleSpace = useCallback(() => {
    clearTimeout(holdTimer.current)
    lastLetterRef.current = null
    const id = nextIdRef.current++
    setChars(prev => [...prev, { id, char: ' ' }])
  }, [])

  const handleBackspace = useCallback(() => {
    clearTimeout(holdTimer.current)
    lastLetterRef.current = null
    setChars(prev => prev.slice(0, -1))
  }, [])

  const handleClear = useCallback(() => {
    clearTimeout(holdTimer.current)
    lastLetterRef.current = null
    setChars([])
    onClear?.()
  }, [onClear])

  // TTS via browser SpeechSynthesis
  const handleSpeak = useCallback(() => {
    if (!sentence.trim()) return
    if (!window.speechSynthesis) {
      alert('Text-to-speech is not supported in this browser.')
      return
    }
    window.speechSynthesis.cancel()
    const utt        = new SpeechSynthesisUtterance(sentence.trim())
    utt.lang         = 'en-US'
    utt.rate         = 0.9
    utt.pitch        = 1.0
    utt.volume       = 1.0
    utt.onstart      = () => setSpeaking(true)
    utt.onend        = () => setSpeaking(false)
    utt.onerror      = () => setSpeaking(false)
    window.speechSynthesis.speak(utt)
  }, [sentence])

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel()
    setSpeaking(false)
  }, [])

  return (
    <section className="sentence-builder" aria-label="Sentence builder">
      <p className="sentence-builder__label">Sentence Builder</p>

      <div
        className="sentence-display"
        role="textbox"
        aria-readonly="true"
        aria-label={`Current sentence: ${sentence || 'empty'}`}
        aria-live="polite"
        aria-atomic="false"
      >
        {chars.length === 0 && (
          <span style={{ color: 'var(--clr-text-dim)', fontWeight: 400, fontSize: '0.9rem' }}>
            Start signing…
          </span>
        )}
        {/* B-SB1 FIX: stable id key — no reconciliation bugs on backspace */}
        {chars.map(({ id, char }) => (
          <span
            key={id}
            className={`sentence-char${char === ' ' ? ' space' : ''}`}
            aria-hidden="true"
          >
            {char === ' ' ? '·' : char}
          </span>
        ))}
        <span className="cursor-blink" aria-hidden="true" />
      </div>

      <div className="sentence-actions">
        <button
          id="btn-space"
          className="btn btn-ghost"
          onClick={handleSpace}
          aria-label="Add space"
          title="Add space"
        >
          ⎵ Space
        </button>

        <button
          id="btn-backspace"
          className="btn btn-ghost"
          onClick={handleBackspace}
          disabled={chars.length === 0}
          aria-label="Delete last character"
          title="Backspace"
        >
          ⌫ Delete
        </button>

        <button
          id="btn-clear"
          className="btn btn-danger"
          onClick={handleClear}
          disabled={chars.length === 0}
          aria-label="Clear sentence"
          title="Clear all"
        >
          🗑 Clear
        </button>

        {speaking ? (
          <button
            id="btn-stop-speak"
            className="btn btn-tts"
            onClick={stopSpeaking}
            aria-label="Stop speaking"
          >
            ⏹ Stop
          </button>
        ) : (
          <button
            id="btn-speak"
            className="btn btn-tts"
            onClick={handleSpeak}
            disabled={!sentence.trim()}
            aria-label="Speak sentence aloud"
            title="Text to speech"
          >
            🔊 Speak
          </button>
        )}
      </div>
    </section>
  )
}
