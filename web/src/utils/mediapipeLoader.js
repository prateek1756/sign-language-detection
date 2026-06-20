/**
 * mediapipeLoader.js
 *
 * Lazy-loads MediaPipe Hands script from a public CDN (jsDelivr) only
 * when requested, preventing global bundle bloat.
 *
 * Returns a Promise resolving to the window.Hands constructor.
 */

let loadingPromise = null

export function loadMediaPipeHands() {
  if (window.Hands) {
    return Promise.resolve(window.Hands)
  }

  if (loadingPromise) {
    return loadingPromise
  }

  loadingPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script")
    script.src = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"
    script.async = true
    script.crossOrigin = "anonymous"

    script.onload = () => {
      if (window.Hands) {
        resolve(window.Hands)
      } else {
        reject(new Error("MediaPipe Hands script loaded, but window.Hands is not defined"))
      }
    }

    script.onerror = () => {
      loadingPromise = null // allow retrying
      reject(new Error("Failed to load MediaPipe Hands script from CDN"))
    }

    document.head.appendChild(script)
  })

  return loadingPromise
}
