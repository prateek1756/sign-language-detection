/**
 * CameraFeed.jsx — Webcam video + MediaPipe landmark skeleton overlay.
 *
 * Renders:
 *  - <video> with mirrored stream
 *  - <canvas> for 21-point hand skeleton drawn in real-time
 *  - Loading skeleton while camera initialises
 *  - Placeholder when camera is off
 *
 * B-CAM1 FIX: canvas dimensions are synced to actual video track size on
 * loadedmetadata, not hardcoded to 640×480. This ensures landmark overlay
 * aligns correctly at any resolution (720p, 1080p, etc.).
 */
import { useEffect, useRef, useCallback } from 'react'

// MediaPipe hand connections (21 landmark indices)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],           // thumb
  [0,5],[5,6],[6,7],[7,8],           // index finger
  [0,9],[9,10],[10,11],[11,12],      // middle finger
  [0,13],[13,14],[14,15],[15,16],    // ring finger
  [0,17],[17,18],[18,19],[19,20],    // pinky
  [5,9],[9,13],[13,17],              // palm
]

const LM_COLOR     = '#7c5cfc'
const LM_TIP_COLOR = '#00d4aa'
const CONN_COLOR   = 'rgba(124, 92, 252, 0.55)'
const TIP_INDICES  = [4, 8, 12, 16, 20] // fingertip landmarks

export default function CameraFeed({ videoRef, isActive, isLoading, landmarks }) {
  const canvasRef = useRef(null)

  // B-CAM1 FIX: sync canvas size to actual video dimensions on metadata load
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onMeta = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width  = video.videoWidth  || 640
      canvas.height = video.videoHeight || 480
    }
    video.addEventListener('loadedmetadata', onMeta)
    return () => video.removeEventListener('loadedmetadata', onMeta)
  }, [videoRef])

  const drawLandmarks = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!landmarks || landmarks.length !== 21) return

    const W = canvas.width
    const H = canvas.height

    // Draw connections first (under dots)
    ctx.strokeStyle = CONN_COLOR
    ctx.lineWidth   = 2
    ctx.lineCap     = 'round'

    for (const [a, b] of HAND_CONNECTIONS) {
      const lmA = landmarks[a]
      const lmB = landmarks[b]
      if (!lmA || !lmB) continue
      ctx.beginPath()
      // landmarks are normalized [0,1] — mirror X because video is mirrored
      ctx.moveTo((1 - lmA.x) * W, lmA.y * H)
      ctx.lineTo((1 - lmB.x) * W, lmB.y * H)
      ctx.stroke()
    }

    // Draw landmark dots
    for (let i = 0; i < landmarks.length; i++) {
      const lm    = landmarks[i]
      const isTip = TIP_INDICES.includes(i)
      const x     = (1 - lm.x) * W
      const y     = lm.y * H
      const r     = isTip ? 6 : 4

      // Glow effect
      ctx.shadowColor = isTip ? LM_TIP_COLOR : LM_COLOR
      ctx.shadowBlur  = isTip ? 12 : 6

      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fillStyle = isTip ? LM_TIP_COLOR : LM_COLOR
      ctx.fill()
    }

    ctx.shadowBlur = 0
  }, [landmarks])

  // Re-draw when landmarks change
  useEffect(() => {
    drawLandmarks()
  }, [drawLandmarks])

  return (
    <div className="camera-feed" id="camera-feed-container">
      {/* Mirrored video */}
      <video
        ref={videoRef}
        id="camera-video"
        muted
        playsInline
        style={{
          transform: 'scaleX(-1)',
          display: isActive ? 'block' : 'none',
        }}
        aria-label="Live camera feed"
      />

      {/* Landmark overlay canvas — dimensions set dynamically via loadedmetadata */}
      <canvas
        ref={canvasRef}
        id="landmark-canvas"
        style={{ display: isActive ? 'block' : 'none' }}
        aria-hidden="true"
      />

      {/* Loading skeleton — shown between Start click and first video frame */}
      {isLoading && !isActive && (
        <div
          className="camera-feed__placeholder"
          role="status"
          aria-label="Camera initialising"
          aria-live="polite"
        >
          <div
            style={{
              width: 48, height: 48, borderRadius: '50%',
              border: '3px solid var(--clr-card-border)',
              borderTopColor: 'var(--clr-accent)',
              animation: 'spin 0.8s linear infinite',
            }}
            aria-hidden="true"
          />
          <p style={{ fontSize: '0.85rem', color: 'var(--clr-text-muted)' }}>
            Starting camera…
          </p>
        </div>
      )}

      {/* Placeholder when camera is fully off */}
      {!isActive && !isLoading && (
        <div className="camera-feed__placeholder" role="img" aria-label="Camera inactive">
          <div className="camera-feed__placeholder-icon" aria-hidden="true">🤟</div>
          <p>Click <strong>Start</strong> to activate camera</p>
        </div>
      )}
    </div>
  )
}
