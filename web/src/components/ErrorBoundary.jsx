/**
 * ErrorBoundary.jsx — Root-level React error boundary.
 *
 * Catches unhandled render errors anywhere in the tree and shows a
 * friendly fallback instead of a blank white screen.
 *
 * Must be a class component — React does not support functional error boundaries.
 */
import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    // In production you'd send this to an error tracking service (e.g. Sentry)
    console.error('[ErrorBoundary] Uncaught error:', error, info.componentStack)
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <div
        role="alert"
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1.5rem',
          background: 'var(--clr-bg, #07070f)',
          color: 'var(--clr-text, #e2e8f0)',
          fontFamily: 'var(--font-sans, system-ui, sans-serif)',
          padding: '2rem',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: '3rem' }} aria-hidden="true">⚠️</div>
        <h1 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0 }}>
          Something went wrong
        </h1>
        <p style={{ fontSize: '0.9rem', color: 'var(--clr-text-muted, #64748b)', maxWidth: 400 }}>
          SignSense AI encountered an unexpected error. Your camera and session
          data have not been saved.
        </p>
        {this.state.error && (
          <pre
            style={{
              fontSize: '0.72rem',
              color: 'var(--clr-red, #ef4444)',
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.2)',
              borderRadius: '8px',
              padding: '0.75rem 1rem',
              maxWidth: 480,
              overflowX: 'auto',
              textAlign: 'left',
            }}
          >
            {this.state.error.message}
          </pre>
        )}
        <button
          onClick={this.handleReload}
          style={{
            padding: '0.5rem 1.5rem',
            background: 'var(--clr-accent, #7c5cfc)',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            fontSize: '0.9rem',
            fontWeight: 600,
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          Reload App
        </button>
      </div>
    )
  }
}
