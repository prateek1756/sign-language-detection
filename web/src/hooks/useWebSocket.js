/**
 * useWebSocket.js — React 19 WebSocket hook with auto-reconnect.
 *
 * Features:
 *  - Exponential back-off reconnect (max 30s)
 *  - Status: 'disconnected' | 'connecting' | 'connected' | 'error'
 *  - sendFrame(b64, mode) — no-op when socket not open (safe to call always)
 *  - Cleanup on unmount
 */
import { useRef, useEffect, useCallback, useState } from 'react'

const WS_URL        = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/stream'
const BASE_DELAY_MS = 1_500
const MAX_DELAY_MS  = 30_000

export function useWebSocket() {
  const wsRef           = useRef(null)
  const reconnectTimer  = useRef(null)
  const attemptRef      = useRef(0)
  const mountedRef      = useRef(true)
  // B-WS1 FIX: track intentional closes so onclose doesn't schedule a reconnect
  const intentionalRef  = useRef(false)

  const [status,     setStatus]     = useState('disconnected')
  const [lastResult, setLastResult] = useState(null)

  const _clearTimer = () => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
  }

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return }
      attemptRef.current = 0   // reset back-off counter on success
      setStatus('connected')
    }

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return
      try {
        const data = JSON.parse(evt.data)
        setLastResult(data)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onerror = () => {
      if (!mountedRef.current) return
      setStatus('error')
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setStatus('disconnected')
      wsRef.current = null

      // B-WS1 FIX: skip reconnect when the user explicitly called disconnect()
      if (intentionalRef.current) {
        intentionalRef.current = false
        return
      }

      // Exponential back-off reconnect
      attemptRef.current += 1
      const delay = Math.min(BASE_DELAY_MS * 2 ** (attemptRef.current - 1), MAX_DELAY_MS)
      reconnectTimer.current = setTimeout(connect, delay)
    }
  }, [])

  const disconnect = useCallback(() => {
    _clearTimer()
    attemptRef.current = 0
    // B-WS1 FIX: mark as intentional before closing so onclose skips reconnect
    intentionalRef.current = true
    wsRef.current?.close()
    wsRef.current = null
    setStatus('disconnected')
  }, [])

  /**
   * Send a frame to the inference backend.
   * @param {string} b64image - base64 JPEG data URL
   * @param {'letter'|'word'} mode
   */
  const sendFrame = useCallback((b64image, mode = 'letter') => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    wsRef.current.send(JSON.stringify({ image: b64image, mode }))
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      _clearTimer()
      wsRef.current?.close()
    }
  }, [])

  return { status, lastResult, connect, disconnect, sendFrame }
}
