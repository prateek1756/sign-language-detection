/**
 * useCamera.js — React 19 custom hook for webcam stream management.
 *
 * Responsibilities:
 *  - Request getUserMedia with ideal 640×480 resolution
 *  - Attach stream to a <video> ref
 *  - Provide a captureFrame() method that returns base64 JPEG
 *  - Handle permissions gracefully with a typed error state
 *  - Expose isLoading state (true between startCamera call and first frame)
 */
import { useRef, useCallback, useState } from 'react'

export function useCamera() {
  const videoRef          = useRef(null)
  const streamRef         = useRef(null)
  const captureCanvasRef  = useRef(null)
  const [isActive,  setIsActive]  = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error,     setError]     = useState(null)

  const startCamera = useCallback(async () => {
    setError(null)
    setIsLoading(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:      { ideal: 640 },
          height:     { ideal: 480 },
          facingMode: 'user',
          frameRate:  { ideal: 30 },
        },
        audio: false,
      })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        // Wait for metadata to load before calling play()
        await new Promise((res, rej) => {
          videoRef.current.onloadedmetadata = res
          videoRef.current.onerror = rej
        })
        await videoRef.current.play()
      }

      // Create an off-screen canvas for JPEG capture
      if (!captureCanvasRef.current) {
        const c = document.createElement('canvas')
        c.width  = 640
        c.height = 480
        captureCanvasRef.current = c
      }

      setIsActive(true)
    } catch (err) {
      const msg =
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access and try again.'
          : err.name === 'NotFoundError'
          ? 'No camera found. Please connect a webcam.'
          : `Camera error: ${err.message}`
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsActive(false)
    setIsLoading(false)
  }, [])

  /**
   * Capture current video frame as base64 JPEG.
   * Returns null if video is not ready yet.
   */
  const captureFrame = useCallback((quality = 0.72) => {
    const video  = videoRef.current
    const canvas = captureCanvasRef.current
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null
    }
    const ctx = canvas.getContext('2d')
    // Mirror the frame horizontally to match the "selfie" expectation
    ctx.save()
    ctx.scale(-1, 1)
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
    ctx.restore()
    return canvas.toDataURL('image/jpeg', quality)
  }, [])

  return { videoRef, isActive, isLoading, error, startCamera, stopCamera, captureFrame }
}
