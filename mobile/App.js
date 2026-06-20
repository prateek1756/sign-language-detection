import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  Dimensions,
  Platform,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const FRAME_INTERVAL_MS = 250; // ~4 FPS - balanced for mobile CPU encoding and network bandwidth

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraActive, setCameraActive] = useState(false);
  const [mode, setMode] = useState('letter'); // 'letter' | 'word'
  const [dialect, setDialect] = useState('ASL');
  const [prediction, setPrediction] = useState(null);
  const [sentence, setSentence] = useState([]);
  const [wsUrl, setWsUrl] = useState('ws://192.168.1.100:8000/ws/stream'); // USER must edit this to match backend host
  const [wsStatus, setWsStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected' | 'error'
  const [showSettings, setShowSettings] = useState(false);

  const cameraRef = useRef(null);
  const wsRef = useRef(null);
  const loopRef = useRef(null);
  const isCapturingRef = useRef(false);
  const lastResultRef = useRef(null);

  // ── Text-to-Speech ──
  const speakSentence = useCallback(() => {
    if (sentence.length === 0) return;
    const text = sentence.join(' ');
    Speech.speak(text, { language: 'en-US' });
  }, [sentence]);

  // ── WebSocket Management ──
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    setWsStatus('connecting');
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus('connected');
        console.log('🔌 Mobile WebSocket Connected');
      };

      ws.onmessage = (evt) => {
        try {
          const res = JSON.parse(evt.data);
          if (res.error) {
            console.warn('Backend API Error:', res.error);
            return;
          }
          setPrediction(res);

          // Handle automatic sentence construction
          const letter = res.letter ?? res.word;
          if (letter && letter !== 'nothing' && !res.buffering) {
            if (letter === 'space') {
              // Commit space
              setSentence(prev => [...prev, ' ']);
            } else if (letter === 'del') {
              // Commit backspace
              setSentence(prev => prev.slice(0, -1));
            } else if (lastResultRef.current !== letter) {
              lastResultRef.current = letter;
              setSentence(prev => {
                if (mode === 'letter') {
                  // Append letter directly to the last word or start a new one
                  if (prev.length === 0) return [letter];
                  const copy = [...prev];
                  const lastIdx = copy.length - 1;
                  copy[lastIdx] = copy[lastIdx] + letter;
                  return copy;
                } else {
                  // Word mode
                  return [...prev, letter];
                }
              });
            }
          } else if (!letter || letter === 'nothing') {
            lastResultRef.current = null;
          }
        } catch (err) {
          console.warn('Failed parsing WS message:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setWsStatus('error');
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        wsRef.current = null;
        console.log('🔌 Mobile WebSocket Disconnected');
      };
    } catch (e) {
      setWsStatus('error');
    }
  }, [wsUrl, mode]);

  const disconnectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsStatus('disconnected');
  }, []);

  // Sync WebSocket state on URL change
  useEffect(() => {
    return () => {
      disconnectWs();
    };
  }, [disconnectWs]);

  // ── Frame Capturing Loop ──
  const captureAndSend = useCallback(async () => {
    if (!cameraActive || isCapturingRef.current) return;
    if (!cameraRef.current) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    isCapturingRef.current = true;
    try {
      const options = {
        quality: 0.15,
        base64: true,
        skipProcessing: true,
      };
      const photo = await cameraRef.current.takePictureAsync(options);
      
      if (photo && photo.base64 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send base64 frame along with current letter/word mode
        wsRef.current.send(
          JSON.stringify({
            image: photo.base64,
            mode: mode,
          })
        );
      }
    } catch (err) {
      console.warn('Frame capture failed:', err);
    } finally {
      isCapturingRef.current = false;
    }
  }, [cameraActive, mode]);

  // Trigger loop interval when camera and connection are both active
  useEffect(() => {
    if (cameraActive && wsStatus === 'connected') {
      loopRef.current = setInterval(captureAndSend, FRAME_INTERVAL_MS);
    } else {
      if (loopRef.current) {
        clearInterval(loopRef.current);
        loopRef.current = null;
      }
    }
    return () => {
      if (loopRef.current) {
        clearInterval(loopRef.current);
        loopRef.current = null;
      }
    };
  }, [cameraActive, wsStatus, captureAndSend]);

  // ── Start / Stop Signing Session ──
  const startSession = async () => {
    if (!permission || !permission.granted) {
      const res = await requestPermission();
      if (!res.granted) return;
    }
    connectWs();
    setCameraActive(true);
  };

  const stopSession = () => {
    setCameraActive(false);
    disconnectWs();
    setPrediction(null);
    lastResultRef.current = null;
  };

  // ── Render Helpers ──
  if (!permission) {
    return (
      <SafeAreaView style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#7c5cfc" />
        <Text style={styles.loadingText}>Initializing camera...</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>SignSense AI</Text>
            <Text style={styles.headerSubtitle}>Real-Time Translation</Text>
          </View>
          <TouchableOpacity
            style={[styles.settingsButton, showSettings && styles.activeTabButton]}
            onPress={() => setShowSettings(!showSettings)}
          >
            <Text style={styles.settingsButtonText}>⚙️ Settings</Text>
          </TouchableOpacity>
        </View>

        {/* Connection Settings */}
        {showSettings && (
          <View style={styles.settingsCard}>
            <Text style={styles.sectionTitle}>API Settings</Text>
            <Text style={styles.label}>Backend WebSocket URL:</Text>
            <TextInput
              style={styles.input}
              value={wsUrl}
              onChangeText={setWsUrl}
              placeholder="ws://192.168.1.X:8000/ws/stream"
              placeholderTextColor="#64748b"
              autoCapitalize="none"
              autoCorrect={false}
            />
            <View style={styles.settingsStatusRow}>
              <Text style={styles.label}>Status:</Text>
              <View style={[styles.statusBadge, styles[wsStatus]]}>
                <Text style={styles.statusBadgeText}>
                  {wsStatus.toUpperCase()}
                </Text>
              </View>
            </View>
            <View style={styles.settingsActions}>
              <TouchableOpacity style={styles.btnSmall} onPress={connectWs}>
                <Text style={styles.btnTextSmall}>Connect</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btnSmall, styles.btnDangerSmall]} onPress={disconnectWs}>
                <Text style={styles.btnTextSmall}>Disconnect</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Camera Container */}
        <View style={styles.cameraCard}>
          {cameraActive ? (
            <View style={styles.cameraWrapper}>
              <CameraView
                ref={cameraRef}
                style={styles.camera}
                facing="front"
              />
              <View style={styles.liveIndicator}>
                <View style={styles.liveDot} />
                <Text style={styles.liveText}>LIVE</Text>
              </View>
            </View>
          ) : (
            <View style={styles.cameraPlaceholder}>
              <Text style={styles.placeholderEmoji}>🤟</Text>
              <Text style={styles.placeholderText}>Webcam offline. Start a session below.</Text>
            </View>
          )}
        </View>

        {/* Mode / Dialect Control bar */}
        <View style={styles.controlBar}>
          <View style={styles.tabGroup}>
            <TouchableOpacity
              style={[styles.tabButton, mode === 'letter' && styles.activeTabButton]}
              onPress={() => setMode('letter')}
            >
              <Text style={[styles.tabButtonText, mode === 'letter' && styles.activeTabButtonText]}>✋ Letter</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.tabButton, mode === 'word' && styles.activeTabButton]}
              onPress={() => setMode('word')}
            >
              <Text style={[styles.tabButtonText, mode === 'word' && styles.activeTabButtonText]}>🌊 Word</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.dialectBadge}>
            <Text style={styles.dialectText}>{dialect}</Text>
          </View>
        </View>

        {/* Start / Stop Sessions */}
        <View style={styles.sessionActions}>
          {!cameraActive ? (
            <TouchableOpacity style={styles.btnPrimary} onPress={startSession}>
              <Text style={styles.btnText}>Start Translation</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[styles.btnPrimary, styles.btnDanger]} onPress={stopSession}>
              <Text style={styles.btnText}>Stop Session</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Prediction Outputs */}
        {cameraActive && (
          <View style={styles.predictionCard}>
            <View style={styles.predictionRow}>
              <View>
                <Text style={styles.labelMicro}>PREDICTION</Text>
                <Text style={styles.predictionText}>
                  {prediction?.buffering ? '✍️ Signing...' : 
                   (prediction?.letter === 'nothing' || prediction?.word === 'nothing' ? '—' : 
                    prediction?.letter === 'space' || prediction?.word === 'space' ? '⎵' : 
                    prediction?.letter === 'del' || prediction?.word === 'del' ? '⌫' : 
                    prediction?.letter ?? prediction?.word ?? '—')}
                </Text>
              </View>
              <View style={styles.latencyContainer}>
                <Text style={styles.labelMicro}>LATENCY</Text>
                <Text style={styles.latencyText}>
                  {prediction?.latency_ms ? `${prediction.latency_ms}ms` : '0ms'}
                </Text>
              </View>
            </View>

            {/* Confidence Bar */}
            {prediction && (prediction.confidence !== undefined || prediction.buffering) && (
              <View style={styles.confidenceContainer}>
                <View style={styles.confidenceBarBg}>
                  <View 
                    style={[
                      styles.confidenceBarFill, 
                      { width: `${(prediction.buffering ? 0.3 : prediction.confidence ?? 0.0) * 100}%` }
                    ]} 
                  />
                </View>
                <Text style={styles.confidenceText}>
                  {prediction.buffering ? 'buffering sequence' : `${Math.round((prediction.confidence ?? 0.0) * 100)}% confidence`}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* Sentence Builder */}
        <View style={styles.sentenceCard}>
          <Text style={styles.labelMicro}>TRANSLATED SENTENCE</Text>
          <View style={styles.sentenceContainer}>
            {sentence.length === 0 ? (
              <Text style={styles.sentencePlaceholder}>Waiting for signs...</Text>
            ) : (
              <Text style={styles.sentenceText}>{sentence.join('')}</Text>
            )}
          </View>
          <View style={styles.sentenceActions}>
            <TouchableOpacity 
              style={[styles.btnAction, sentence.length === 0 && styles.disabledButton]} 
              onPress={speakSentence}
              disabled={sentence.length === 0}
            >
              <Text style={styles.btnActionText}>🔊 Speak</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.btnAction, sentence.length === 0 && styles.disabledButton]} 
              onPress={() => setSentence(prev => prev.slice(0, -1))}
              disabled={sentence.length === 0}
            >
              <Text style={styles.btnActionText}>⌫ Delete</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.btnAction, sentence.length === 0 && styles.disabledButton]} 
              onPress={() => setSentence([])}
              disabled={sentence.length === 0}
            >
              <Text style={styles.btnActionText}>🗑️ Clear</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#11111e',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#11111e',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  loadingText: {
    color: '#64748b',
    fontSize: 14,
  },
  scrollContainer: {
    padding: 16,
    gap: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 2,
  },
  settingsButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 8,
    paddingVertical: 6,
    paddingHorizontal: 12,
  },
  settingsButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  settingsCard: {
    backgroundColor: '#1a1a2e',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 12,
    padding: 16,
    gap: 12,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#7c5cfc',
    marginBottom: 4,
  },
  label: {
    fontSize: 12,
    color: '#94a3b8',
  },
  input: {
    backgroundColor: '#11111e',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 8,
    color: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 12,
    fontSize: 13,
  },
  settingsStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  statusBadge: {
    borderRadius: 16,
    paddingVertical: 4,
    paddingHorizontal: 10,
    alignSelf: 'flex-start',
  },
  statusBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    color: '#fff',
  },
  disconnected: {
    backgroundColor: '#ef4444',
  },
  connecting: {
    backgroundColor: '#eab308',
  },
  connected: {
    backgroundColor: '#00d4aa',
  },
  error: {
    backgroundColor: '#ef4444',
  },
  settingsActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 8,
  },
  btnSmall: {
    flex: 1,
    backgroundColor: '#7c5cfc',
    borderRadius: 8,
    paddingVertical: 8,
    alignItems: 'center',
  },
  btnDangerSmall: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderWidth: 1,
    borderColor: '#ef4444',
  },
  btnTextSmall: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  cameraCard: {
    width: '100%',
    aspectRatio: 4 / 3,
    backgroundColor: '#090910',
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  cameraWrapper: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  liveIndicator: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: 'rgba(0, 212, 170, 0.15)',
    borderWidth: 1,
    borderColor: '#00d4aa',
    borderRadius: 99,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 3,
    paddingHorizontal: 8,
    gap: 4,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#00d4aa',
  },
  liveText: {
    fontSize: 9,
    fontWeight: '800',
    color: '#00d4aa',
  },
  cameraPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  placeholderEmoji: {
    fontSize: 48,
    opacity: 0.5,
  },
  placeholderText: {
    color: '#64748b',
    fontSize: 13,
    textAlign: 'center',
    paddingHorizontal: 30,
  },
  controlBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  tabGroup: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 10,
    padding: 3,
    gap: 2,
  },
  tabButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  activeTabButton: {
    backgroundColor: '#7c5cfc',
  },
  tabButtonText: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '700',
  },
  activeTabButtonText: {
    color: '#fff',
  },
  dialectBadge: {
    backgroundColor: 'rgba(124, 92, 252, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(124, 92, 252, 0.2)',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  dialectText: {
    color: '#7c5cfc',
    fontSize: 12,
    fontWeight: '700',
  },
  sessionActions: {
    width: '100%',
  },
  btnPrimary: {
    backgroundColor: '#7c5cfc',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    shadowColor: '#7c5cfc',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  btnDanger: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderWidth: 1,
    borderColor: '#ef4444',
    shadowColor: '#ef4444',
    elevation: 0,
  },
  btnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
  predictionCard: {
    backgroundColor: '#1a1a2e',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 16,
    padding: 16,
    gap: 12,
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  labelMicro: {
    fontSize: 9,
    fontWeight: '700',
    color: '#64748b',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  predictionText: {
    fontSize: 22,
    fontWeight: '800',
    color: '#7c5cfc',
    marginTop: 4,
  },
  latencyContainer: {
    alignItems: 'flex-end',
  },
  latencyText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
    marginTop: 4,
  },
  confidenceContainer: {
    gap: 6,
  },
  confidenceBarBg: {
    width: '100%',
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceBarFill: {
    height: '100%',
    backgroundColor: '#00d4aa',
    borderRadius: 3,
  },
  confidenceText: {
    fontSize: 11,
    color: '#64748b',
  },
  sentenceCard: {
    backgroundColor: '#1a1a2e',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 16,
    padding: 16,
    gap: 12,
  },
  sentenceContainer: {
    minHeight: 64,
    backgroundColor: '#11111e',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    padding: 12,
    justifyContent: 'center',
  },
  sentencePlaceholder: {
    color: '#64748b',
    fontSize: 14,
    fontStyle: 'italic',
  },
  sentenceText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    lineHeight: 22,
  },
  sentenceActions: {
    flexDirection: 'row',
    gap: 8,
  },
  btnAction: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
  },
  btnActionText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  disabledButton: {
    opacity: 0.4,
  },
});
