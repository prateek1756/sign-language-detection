/**
 * useEdgeInference.js
 *
 * React hook to manage in-browser neural network inference using ONNX Runtime Web.
 * Loads MLP and LSTM models from CDN/public folder, feeds MediaPipe landmarks,
 * and performs real-time classification with smoothing.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import * as ort from 'onnxruntime-web'

// Config constants
const CONFIDENCE_THRESHOLD = 0.55
const SMOOTHING_WINDOW     = 4
const SEQUENCE_LEN         = 30

const ASL_CLASSES = [
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "space", "del", "nothing"
]

// ── Helpers ──

class TemporalSmoother {
  constructor(windowSize = SMOOTHING_WINDOW) {
    this.windowSize = windowSize
    this.history = []
  }
  push(labelIdx) {
    this.history.push(labelIdx)
    if (this.history.length > this.windowSize) {
      this.history.shift()
    }
    const counts = {}
    let maxCount = 0
    let winner = labelIdx
    for (const val of this.history) {
      counts[val] = (counts[val] || 0) + 1
      if (counts[val] > maxCount) {
        maxCount = counts[val]
        winner = val
      }
    }
    return winner
  }
  reset() {
    this.history = []
  }
  get filled() {
    return this.history.length === this.windowSize
  }
}

class ModeDetector {
  constructor() {
    this.state = 'letter' // 'letter' or 'word'
    this.windowSize = 8
    this.startThreshold = 0.035
    this.endThreshold = 0.015
    this.consecutiveFrames = 3
    this.motion = []
    this.prevTips = null
    this.stopCounter = 0
  }
  update(landmarks) {
    // Slice fingertips (indices 4, 8, 12, 16, 20)
    const tips = [4, 8, 12, 16, 20].map(idx => [
      landmarks[idx * 3],
      landmarks[idx * 3 + 1],
      landmarks[idx * 3 + 2]
    ])

    if (this.prevTips !== null) {
      let sumDisp = 0
      for (let i = 0; i < 5; i++) {
        const dx = tips[i][0] - this.prevTips[i][0]
        const dy = tips[i][1] - this.prevTips[i][1]
        const dz = tips[i][2] - this.prevTips[i][2]
        sumDisp += Math.sqrt(dx*dx + dy*dy + dz*dz)
      }
      const meanDisp = sumDisp / 5
      this.motion.push(meanDisp)
      if (this.motion.length > this.windowSize) {
        this.motion.shift()
      }
    }
    this.prevTips = tips

    if (this.motion.length < 3) {
      return this.state
    }

    const sum = this.motion.reduce((a, b) => a + b, 0)
    const meanMotion = sum / this.motion.length

    if (this.state === 'letter') {
      if (meanMotion > this.startThreshold) {
        this.state = 'word'
        this.stopCounter = 0
      }
    } else { // state is 'word'
      if (meanMotion < this.endThreshold) {
        this.stopCounter++
        if (this.stopCounter >= this.consecutiveFrames) {
          this.state = 'letter'
        }
      } else {
        this.stopCounter = 0
      }
    }

    return this.state
  }
  reset() {
    this.state = 'letter'
    this.motion = []
    this.prevTips = null
    this.stopCounter = 0
  }
}

function alignPalmPlane(landmarks) {
  // landmarks is a flat array/Float32Array of length 63 representing 21 landmarks (x, y, z)
  const lms = []
  for (let i = 0; i < 21; i++) {
    lms.push([landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]])
  }

  const p5 = lms[5]
  const p17 = lms[17]

  // Y-axis (from wrist 0 to index MCP 5)
  const normY = Math.sqrt(p5[0]*p5[0] + p5[1]*p5[1] + p5[2]*p5[2])
  const yAxis = p5.map(val => val / (normY + 1e-8))

  // Temp vector for Pinky MCP 17
  const norm17 = Math.sqrt(p17[0]*p17[0] + p17[1]*p17[1] + p17[2]*p17[2])
  const vTmp = p17.map(val => val / (norm17 + 1e-8))

  // Z-axis (palm normal): cross product of yAxis and vTmp
  const zRaw = [
    yAxis[1] * vTmp[2] - yAxis[2] * vTmp[1],
    yAxis[2] * vTmp[0] - yAxis[0] * vTmp[2],
    yAxis[0] * vTmp[1] - yAxis[1] * vTmp[0]
  ]
  const normZ = Math.sqrt(zRaw[0]*zRaw[0] + zRaw[1]*zRaw[1] + zRaw[2]*zRaw[2])
  const zAxis = zRaw.map(val => val / (normZ + 1e-8))

  // X-axis (transverse): cross product of yAxis and zAxis
  const xRaw = [
    yAxis[1] * zAxis[2] - yAxis[2] * zAxis[1],
    yAxis[2] * zAxis[0] - yAxis[0] * zAxis[2],
    yAxis[0] * zAxis[1] - yAxis[1] * zAxis[0]
  ]
  const normX = Math.sqrt(xRaw[0]*xRaw[0] + xRaw[1]*xRaw[1] + xRaw[2]*xRaw[2])
  const xAxis = xRaw.map(val => val / (normX + 1e-8))

  // Rotate landmarks: rotated = landmarks @ R
  const rotated = new Float32Array(63)
  for (let i = 0; i < 21; i++) {
    const p = lms[i]
    rotated[i * 3] = p[0] * xAxis[0] + p[1] * xAxis[1] + p[2] * xAxis[2]
    rotated[i * 3 + 1] = p[0] * yAxis[0] + p[1] * yAxis[1] + p[2] * yAxis[2]
    rotated[i * 3 + 2] = p[0] * zAxis[0] + p[1] * zAxis[1] + p[2] * zAxis[2]
  }

  return rotated
}

function computeGeometricFeatures(landmarks) {
  // landmarks is a flat array of size 63 (normalized and rotated)
  const lms = []
  for (let i = 0; i < 21; i++) {
    lms.push([landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]])
  }

  const features = []

  // 1. Fingertip-to-wrist distances (indices 4, 8, 12, 16, 20)
  const tips = [4, 8, 12, 16, 20]
  for (const tip of tips) {
    const p = lms[tip]
    const dist = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    features.push(dist)
  }

  // 2. Tip-to-Tip distances
  for (let i = 0; i < tips.length - 1; i++) {
    const p1 = lms[tips[i]]
    const p2 = lms[tips[i+1]]
    const dx = p1[0] - p2[0]
    const dy = p1[1] - p2[1]
    const dz = p1[2] - p2[2]
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz)
    features.push(dist)
  }

  // 3. Joint Bending Angles (5 features)
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
  const norm = (a) => Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

  const fingerVectors = [
    [sub(lms[2], lms[1]), sub(lms[4], lms[3])], // Thumb
    [sub(lms[6], lms[5]), sub(lms[8], lms[7])], // Index
    [sub(lms[10], lms[9]), sub(lms[12], lms[11])], // Middle
    [sub(lms[14], lms[13]), sub(lms[16], lms[15])], // Ring
    [sub(lms[18], lms[17]), sub(lms[20], lms[19])] // Pinky
  ]

  for (const [v1, v2] of fingerVectors) {
    const n1 = norm(v1)
    const n2 = norm(v2)
    if (n1 < 1e-8 || n2 < 1e-8) {
      features.push(0.0)
    } else {
      let cosine = dot(v1, v2) / (n1 * n2)
      cosine = Math.max(-1.0, Math.min(1.0, cosine))
      const angle = Math.acos(cosine)
      features.push(angle)
    }
  }

  return new Float32Array(features)
}

export function useEdgeInference(options = {}) {
  const { alignRotation = false, getGeometric = false } = options
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const mlpSessionRef  = useRef(null)
  const lstmSessionRef = useRef(null)

  const smootherRef    = useRef(new TemporalSmoother())
  const modeDetRef     = useRef(new ModeDetector())
  const seqBufferRef   = useRef([])

  // Load models at mount
  useEffect(() => {
    async function initSessions() {
      setLoading(true)
      setError(null)
      try {
        const mlpUrl  = "/models/asl_mlp.onnx"
        const lstmUrl = "/models/asl_lstm.onnx"

        const [mlpSession, lstmSession] = await Promise.all([
          ort.InferenceSession.create(mlpUrl,  { executionProviders: ['webgl', 'wasm'] }),
          ort.InferenceSession.create(lstmUrl, { executionProviders: ['webgl', 'wasm'] })
        ])

        mlpSessionRef.current  = mlpSession
        lstmSessionRef.current = lstmSession
        setLoading(false)
      } catch (err) {
        setError(`Failed to load ONNX model sessions: ${err.message}`)
        setLoading(false)
      }
    }
    initSessions()
  }, [])

  const resetState = useCallback(() => {
    smootherRef.current.reset()
    modeDetRef.current.reset()
    seqBufferRef.current = []
  }, [])

  /**
   * Run client-side inference on a single 63-dim landmark array.
   */
  const predictFrame = useCallback(async (landmarks) => {
    const mlp = mlpSessionRef.current
    if (!mlp) return { error: "MLP model not loaded" }

    const t0 = performance.now()
    try {
      const prevMode = modeDetRef.current.state
      const mode = modeDetRef.current.update(landmarks)
      const gestureEnded = (prevMode === 'word' && mode === 'letter')

      // Apply palm plane rotation alignment
      let processed = landmarks
      if (alignRotation) {
        processed = alignPalmPlane(processed)
      }

      // Compute and append geometric features
      if (getGeometric) {
        const geom = computeGeometricFeatures(processed)
        const combined = new Float32Array(processed.length + geom.length)
        combined.set(processed)
        combined.set(geom, processed.length)
        processed = combined
      }

      const inputDim = processed.length
      const tensorInput = new ort.Tensor('float32', new Float32Array(processed), [1, inputDim])
      const feeds = { [mlp.inputNames[0]]: tensorInput }
      const results = await mlp.run(feeds)
      const proba = results[mlp.outputNames[0]].data // Float32Array (29,)

      // Argmax
      let bestIdx = 0
      let bestConf = 0
      for (let i = 0; i < proba.length; i++) {
        if (proba[i] > bestConf) {
          bestConf = proba[i]
          bestIdx = i
        }
      }

      const latencyMs = performance.now() - t0

      // Confidence Gate
      if (bestConf < CONFIDENCE_THRESHOLD) {
        return {
          letter: null,
          confidence: bestConf,
          mode,
          gestureEnded,
          below_threshold: true,
          latency_ms: Math.round(latencyMs * 10) / 10
        }
      }

      // Temporal Smoothing
      const smoothedIdx = smootherRef.current.push(bestIdx)
      const smoothedConf = proba[smoothedIdx]

      // Accumulate for LSTM word sequences
      seqBufferRef.current.push(processed)
      if (seqBufferRef.current.length > SEQUENCE_LEN) {
        seqBufferRef.current.shift()
      }

      // Generate top 3
      const indexed = Array.from(proba).map((c, i) => ({ label: ASL_CLASSES[i], confidence: c }))
      indexed.sort((a, b) => b.confidence - a.confidence)
      const top3 = indexed.slice(0, 3)

      return {
        letter: ASL_CLASSES[smoothedIdx],
        raw_letter: ASL_CLASSES[bestIdx],
        confidence: Math.round(smoothedConf * 10000) / 10000,
        top3,
        mode,
        gestureEnded,
        smoothed: smootherRef.current.filled,
        below_threshold: false,
        latency_ms: Math.round(latencyMs * 10) / 10
      }
    } catch (err) {
      return { error: `Inference failed: ${err.message}` }
    }
  }, [alignRotation, getGeometric])

  /**
   * Run client-side inference on a buffered sequence for LSTM word prediction.
   */
  const predictSequence = useCallback(async (customSequence = null) => {
    const lstm = lstmSessionRef.current
    if (!lstm) return { error: "LSTM model not loaded" }

    const t0 = performance.now()
    try {
      let sequence = customSequence || seqBufferRef.current

      if (sequence.length < 5) {
        return { error: "Too few frames for sequence prediction" }
      }

      // Preprocess custom sequence if supplied
      if (customSequence) {
        sequence = customSequence.map(item => {
          let processed = item
          if (alignRotation) {
            processed = alignPalmPlane(processed)
          }
          if (getGeometric) {
            const geom = computeGeometricFeatures(processed)
            const combined = new Float32Array(processed.length + geom.length)
            combined.set(processed)
            combined.set(geom, processed.length)
            processed = combined
          }
          return processed
        })
      }

      // Pad to SEQUENCE_LEN if needed
      if (sequence.length < SEQUENCE_LEN) {
        const last = sequence[sequence.length - 1]
        const pad = Array(SEQUENCE_LEN - sequence.length).fill(last)
        sequence = [...sequence, ...pad]
      } else if (sequence.length > SEQUENCE_LEN) {
        sequence = sequence.slice(0, SEQUENCE_LEN)
      }

      const featureDim = sequence[0].length
      // Flatten sequence
      const flatArray = new Float32Array(SEQUENCE_LEN * featureDim)
      for (let i = 0; i < SEQUENCE_LEN; i++) {
        flatArray.set(sequence[i], i * featureDim)
      }

      const tensorInput = new ort.Tensor('float32', flatArray, [1, SEQUENCE_LEN, featureDim])
      const feeds = { [lstm.inputNames[0]]: tensorInput }
      const results = await lstm.run(feeds)
      const proba = results[lstm.outputNames[0]].data // Float32Array (29,)

      let bestIdx = 0
      let bestConf = 0
      for (let i = 0; i < proba.length; i++) {
        if (proba[i] > bestConf) {
          bestConf = proba[i]
          bestIdx = i
        }
      }

      const latencyMs = performance.now() - t0

      const indexed = Array.from(proba).map((c, i) => ({ label: ASL_CLASSES[i], confidence: c }))
      indexed.sort((a, b) => b.confidence - a.confidence)
      const top3 = indexed.slice(0, 3)

      return {
        word: bestConf >= CONFIDENCE_THRESHOLD ? ASL_CLASSES[bestIdx] : null,
        confidence: Math.round(bestConf * 10000) / 10000,
        top3,
        frames_used: sequence.length,
        below_threshold: bestConf < CONFIDENCE_THRESHOLD,
        latency_ms: Math.round(latencyMs * 10) / 10
      }
    } catch (err) {
      return { error: `Sequence inference failed: ${err.message}` }
    }
  }, [alignRotation, getGeometric])

  return { loading, error, predictFrame, predictSequence, resetState, seqBuffer: seqBufferRef.current }
}
