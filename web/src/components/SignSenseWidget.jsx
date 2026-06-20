/**
 * SignSenseWidget.jsx
 *
 * Self-contained embeddable B2B SaaS widget for sign language translation.
 * Handles webcam stream, dynamic MediaPipe load, canvas overlays, and
 * client-side edge neural network classification.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { loadMediaPipeHands } from '../utils/mediapipeLoader.js'
import { useEdgeInference } from '../hooks/useEdgeInference.js'

// Skeleton connections mapping for drawing landmarks
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],           // thumb
  [0,5],[5,6],[6,7],[7,8],           // index finger
  [0,9],[9,10],[10,11],[11,12],      // middle finger
  [0,13],[13,14],[14,15],[15,16],    // ring finger
  [0,17],[17,18],[18,19],[19,20],    // pinky
  [5,9],[9,13],[13,17],              // palm
]

const TIP_INDICES  = [4, 8, 12, 16, 20]
const FRAME_INTERVAL_MS = 160 // ~6 FPS

export default function SignSenseWidget({
  mode = 'letter', // 'letter' | 'word'
  width = 640,
  height = 480,
  onPrediction = null,
  theme = 'dark' // 'dark' | 'light'
}) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const [handsLoaded, setHandsLoaded] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [prediction, setPrediction] = useState(null)

  const { loading: modelsLoading, error: modelsError, predictFrame, predictSequence, resetState } = useEdgeInference()

  // Sync models loading error to widget error message state
  useEffect(() => {
    if (modelsError) {
      setErrorMsg(modelsError)
    }
  }, [modelsError])

  // MediaPipe hands singleton tracker instance
  const handsTrackerRef = useRef(null)
  const onResultsRef = useRef(null)

  // Cache volatile prop/state dependencies using refs to prevent onResults reconstruction
  const onPredictionRef = useRef(onPrediction)
  const modeRef = useRef(mode)
  const predictFrameRef = useRef(predictFrame)
  const predictSequenceRef = useRef(predictSequence)

  useEffect(() => {
    onPredictionRef.current = onPrediction
    modeRef.current = mode
    predictFrameRef.current = predictFrame
    predictSequenceRef.current = predictSequence
  }, [onPrediction, mode, predictFrame, predictSequence])

  // ── Theme Setup ──
  const isDark = theme === 'dark'
  const clrBg = isDark ? '#11111e' : '#f8fafc'
  const clrSurface = isDark ? '#1a1a2e' : '#ffffff'
  const clrBorder = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'
  const clrText = isDark ? '#f1f5f9' : '#0f172a'
  const clrMuted = isDark ? '#64748b' : '#94a3b8'
  const clrAccent = '#7c5cfc'

  // ── Load MediaPipe ──
  useEffect(() => {
    loadMediaPipeHands()
      .then((HandsConstructor) => {
        const tracker = new HandsConstructor({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        })
        tracker.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        })
        tracker.onResults((results) => {
          if (onResultsRef.current) {
            onResultsRef.current(results)
          }
        })
        handsTrackerRef.current = tracker
        setHandsLoaded(true)
      })
      .catch((err) => {
        setErrorMsg(`Failed to load MediaPipe Hands: ${err.message}`)
      })

    return () => {
      if (handsTrackerRef.current) {
        handsTrackerRef.current.close()
      }
    }
  }, [])

  // ── Start/Stop Webcam ──
  const stopWebcam = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
    setPrediction(null)
    resetState()
  }, [resetState])

  const startWebcam = useCallback(async () => {
    setErrorMsg(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: width }, height: { ideal: height }, facingMode: 'user' },
        audio: false
      })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth || width
            canvasRef.current.height = videoRef.current.videoHeight || height
          }
        }
        await videoRef.current.play()
      }
      setCameraActive(true)
    } catch (err) {
      setErrorMsg(err.name === 'NotAllowedError' ? 'Camera access permission denied' : `Camera init failed: ${err.message}`)
    }
  }, [width, height])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    }
  }, [])

  // ── Preprocessing & Draw ──
  const drawLandmarks = useCallback((rawLms) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!rawLms || rawLms.length !== 21) return

    const W = canvas.width
    const H = canvas.height

    // Draw lines
    ctx.strokeStyle = 'rgba(124, 92, 252, 0.6)'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    for (const [a, b] of HAND_CONNECTIONS) {
      const lmA = rawLms[a]
      const lmB = rawLms[b]
      if (!lmA || !lmB) continue
      ctx.beginPath()
      ctx.moveTo((1 - lmA.x) * W, lmA.y * H)
      ctx.lineTo((1 - lmB.x) * W, lmB.y * H)
      ctx.stroke()
    }

    // Draw dots
    for (let i = 0; i < rawLms.length; i++) {
      const lm = rawLms[i]
      const isTip = TIP_INDICES.includes(i)
      const x = (1 - lm.x) * W
      const y = lm.y * H
      const r = isTip ? 6 : 4

      ctx.shadowColor = isTip ? '#00d4aa' : '#7c5cfc'
      ctx.shadowBlur = isTip ? 12 : 6
      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fillStyle = isTip ? '#00d4aa' : '#7c5cfc'
      ctx.fill()
    }
    ctx.shadowBlur = 0
  }, [])

  // ── results callback ──
  const onResults = useCallback(async (results) => {
    let extractedLandmarks = null
    let rawLandmarksList = null

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0]
      rawLandmarksList = hand
      
      const aspect = videoRef.current ? videoRef.current.videoWidth / videoRef.current.videoHeight : 1.33
      const rawArray = new Float32Array(63)
      for (let i = 0; i < 21; i++) {
        rawArray[i * 3] = hand[i].x * aspect
        rawArray[i * 3 + 1] = hand[i].y
        rawArray[i * 3 + 2] = hand[i].z
      }
      
      const wristX = rawArray[0]
      const wristY = rawArray[1]
      const wristZ = rawArray[2]
      
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
      const wristCentered = new Float32Array(63)
      for (let i = 0; i < 21; i++) {
        wristCentered[i * 3] = rawArray[i * 3] - wristX
        wristCentered[i * 3 + 1] = rawArray[i * 3 + 1] - wristY
        wristCentered[i * 3 + 2] = rawArray[i * 3 + 2] - wristZ
        
        minX = Math.min(minX, wristCentered[i * 3])
        minY = Math.min(minY, wristCentered[i * 3 + 1])
        maxX = Math.max(maxX, wristCentered[i * 3])
        maxY = Math.max(maxY, wristCentered[i * 3 + 1])
      }

      const sizeX = maxX - minX
      const sizeY = maxY - minY
      const diagonal = Math.sqrt(sizeX * sizeX + sizeY * sizeY)

      if (diagonal > 1e-6) {
        for (let i = 0; i < 63; i++) {
          wristCentered[i] /= diagonal
        }
        extractedLandmarks = wristCentered
      }
    }

    if (rawLandmarksList) {
      drawLandmarks(rawLandmarksList)
    } else {
      const canvas = canvasRef.current
      if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    }

    if (extractedLandmarks) {
      const currentMode = modeRef.current
      const currentPredictFrame = predictFrameRef.current
      const currentPredictSequence = predictSequenceRef.current
      const currentOnPrediction = onPredictionRef.current

      if (currentMode === 'letter') {
        const res = await currentPredictFrame(extractedLandmarks)
        if (res && !res.error) {
          setPrediction(res)
          if (currentOnPrediction && res.letter) {
            currentOnPrediction(res.letter, res.confidence)
          }
        }
      } else {
        // Word mode: predict frame to update motion state
        const res = await currentPredictFrame(extractedLandmarks)
        if (res && res.gestureEnded) {
          const wordRes = await currentPredictSequence()
          if (wordRes && !wordRes.error) {
            setPrediction({ ...wordRes, mode: 'word', gestureEnded: true })
            if (currentOnPrediction && wordRes.word) {
              currentOnPrediction(wordRes.word, wordRes.confidence)
            }
          }
        } else if (res && res.mode === 'word') {
          // Hand is moving
          setPrediction({
            buffering: true,
            mode: 'word',
            latency_ms: res.latency_ms
          })
        } else {
          setPrediction(res) // Static hand pose in word mode
        }
      }
    } else {
      setPrediction(null)
    }
  }, [drawLandmarks])

  // Synchronize onResults to the mutable Ref redirect shell
  useEffect(() => {
    onResultsRef.current = onResults
  }, [onResults])

  // ── Frame Loop ──
  const processFrame = useCallback(async () => {
    const video = videoRef.current
    const tracker = handsTrackerRef.current
    if (!video || !tracker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return

    // Feed to MediaPipe Hands and trigger callback via tracker
    await tracker.send({ image: video })
  }, [])

  // Trigger loop interval when camera goes active
  useEffect(() => {
    if (cameraActive && handsLoaded) {
      intervalRef.current = setInterval(processFrame, FRAME_INTERVAL_MS)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [cameraActive, handsLoaded, processFrame])

  // ── Render ──
  const isWidgetLoading = !handsLoaded || modelsLoading

  return (
    <div
      style={{
        width,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        background: clrBg,
        color: clrText,
        borderRadius: 16,
        border: `1px solid ${clrBorder}`,
        padding: 16,
        boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        boxSizing: 'border-box'
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', color: clrMuted }}>
          🤟 SIGNSENSE EDGE WIDGET
        </span>
        <span
          style={{
            fontSize: '0.7rem',
            padding: '2px 8px',
            borderRadius: 99,
            background: mode === 'word' ? 'rgba(0,212,170,0.12)' : 'rgba(124,92,252,0.12)',
            color: mode === 'word' ? '#00d4aa' : clrAccent,
            fontWeight: 700,
            textTransform: 'uppercase'
          }}
        >
          {mode} mode
        </span>
      </div>

      {/* Video Sandbox */}
      <div
        style={{
          width: '100%',
          height: height - 60,
          background: isDark ? '#090910' : '#e2e8f0',
          borderRadius: 12,
          overflow: 'hidden',
          position: 'relative',
          border: `1px solid ${clrBorder}`
        }}
      >
        <video
          ref={videoRef}
          muted
          playsInline
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scaleX(-1)',
            display: cameraActive ? 'block' : 'none'
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            display: cameraActive ? 'block' : 'none'
          }}
        />

        {/* Loading Spinner */}
        {isWidgetLoading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, background: clrBg }}>
            <div style={{ width: 36, height: 36, borderRadius: '50%', border: `3px solid ${clrBorder}`, borderTopColor: clrAccent, animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: '0.8rem', color: clrMuted }}>Loading Edge AI Engine...</span>
          </div>
        )}

        {/* Inactive Overlay */}
        {!cameraActive && !isWidgetLoading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
            <span style={{ fontSize: '3rem', opacity: 0.6 }}>🤟</span>
            <button
              onClick={startWebcam}
              style={{
                background: clrAccent,
                color: '#fff',
                border: 'none',
                padding: '8px 18px',
                borderRadius: 8,
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
                boxShadow: '0 4px 12px rgba(124,92,252,0.3)'
              }}
            >
              Start Webcam
            </button>
          </div>
        )}

        {/* Error message */}
        {errorMsg && (
          <div style={{ position: 'absolute', bottom: 12, left: 12, right: 12, padding: '8px 12px', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: 8, fontSize: '0.78rem', color: '#ef4444' }}>
            ⚠️ {errorMsg}
          </div>
        )}
      </div>

      {/* Control Bar & Prediction Output */}
      {cameraActive && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: clrSurface, padding: '10px 14px', borderRadius: 10, border: `1px solid ${clrBorder}` }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '0.65rem', color: clrMuted, fontWeight: 600 }}>PREDICTION</span>
            <span style={{ fontSize: '1.2rem', fontWeight: 800, color: prediction?.letter || prediction?.word || prediction?.buffering ? clrAccent : clrMuted }}>
              {prediction?.buffering ? '✍️ Signing...' : 
               (prediction?.letter === 'nothing' ? '—' : prediction?.letter === 'space' ? '⎵' : prediction?.letter === 'del' ? '⌫' : prediction?.letter || prediction?.word || '—')}
            </span>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
            <span style={{ fontSize: '0.65rem', color: clrMuted, fontWeight: 600 }}>LATENCY</span>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: clrText }}>
              {prediction?.latency_ms ? `${prediction.latency_ms}ms` : '0ms'}
            </span>
          </div>

          <button
            onClick={stopWebcam}
            style={{
              background: 'rgba(239,68,68,0.12)',
              color: '#ef4444',
              border: '1px solid #ef4444',
              padding: '6px 12px',
              borderRadius: 6,
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Stop
          </button>
        </div>
      )}

      {/* CSS Animation injection */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
