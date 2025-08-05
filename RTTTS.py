#!/usr/bin/env python3
"""
Real-Time Text-to-Speech using Piper
Handles streaming and immediate speech synthesis
"""

import io
import os
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import queue
from piper import PiperVoice
from threading import Event, Lock
import time

# Global state
voice = None
tts_queue = queue.Queue()
tts_thread = None
tts_active = Event()
voice_lock = Lock()

# Speech streaming state
streaming_active = False
streaming_lock = Lock()

def initialize_voice():
    """Initialize Piper voice model once"""
    global voice
    if voice is None:
        with voice_lock:
            if voice is None:
                try:
                    print("Loading Piper voice model...")
                    voice = PiperVoice.load(
                        model_path="voices/en_US-kathleen-low.onnx",
                        config_path="voices/en_US-kathleen-low.onnx.json"
                    )
                    print("Piper voice model loaded successfully")
                except Exception as e:
                    print(f"Failed to load Piper voice: {e}")
                    voice = None

def _synthesize_audio(text: str) -> tuple:
    """Internal function to synthesize audio data"""
    if not voice:
        return None, None
    
    try:
        # Collect all audio data from AudioChunks
        all_audio_data = []
        sample_rate = None
        
        for audio_chunk in voice.synthesize(text):
            # Use the audio_float_array attribute for best quality
            if hasattr(audio_chunk, 'audio_float_array'):
                all_audio_data.append(audio_chunk.audio_float_array)
                if sample_rate is None:
                    sample_rate = audio_chunk.sample_rate
            else:
                # Fallback to int16 bytes
                if hasattr(audio_chunk, 'audio_int16_bytes'):
                    audio_array = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32767.0
                    all_audio_data.append(audio_float)
                    if sample_rate is None:
                        sample_rate = audio_chunk.sample_rate
                else:
                    continue
        
        if not all_audio_data:
            return None, None
            
        # Concatenate all audio arrays
        complete_audio = np.concatenate(all_audio_data)
        return complete_audio, sample_rate
        
    except Exception as e:
        print(f"[TTS ERROR] Synthesis failed: {e}")
        return None, None

def speak(text: str) -> bool:
    """Synchronous speech synthesis and playback"""
    if not text.strip():
        return False
    
    initialize_voice()
    if not voice:
        print("[TTS ERROR] Voice model not available")
        return False
    
    try:
        audio_data, sample_rate = _synthesize_audio(text)
        if audio_data is None:
            return False
        
        # Play the audio
        sd.play(audio_data, sample_rate)
        sd.wait()
        
        return True
        
    except Exception as e:
        print(f"[TTS ERROR] Playback failed: {e}")
        return False

def _tts_worker():
    """Background worker thread for TTS queue processing"""
    while tts_active.is_set():
        try:
            # Get text from queue with timeout
            text = tts_queue.get(timeout=0.5)
            if text is None:  # Shutdown signal
                break
                
            # Synthesize and play
            speak(text)
            tts_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS WORKER ERROR] {e}")

def start_tts_worker():
    """Start the background TTS worker thread"""
    global tts_thread
    if tts_thread is None or not tts_thread.is_alive():
        tts_active.set()
        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()

def stop_tts_worker():
    """Stop the background TTS worker thread"""
    global tts_thread
    tts_active.clear()
    tts_queue.put(None)  # Shutdown signal
    if tts_thread and tts_thread.is_alive():
        tts_thread.join(timeout=2.0)
    tts_thread = None

def say_async(text: str):
    """Add text to TTS queue for asynchronous playback"""
    if not text.strip():
        return
    
    start_tts_worker()
    try:
        tts_queue.put_nowait(text)
    except queue.Full:
        print("[TTS WARNING] Queue full, dropping text")

# Streaming TTS functions for ollama_client compatibility
def stream_speech():
    """Initialize speech streaming (prepare for chunked speech)"""
    global streaming_active
    with streaming_lock:
        streaming_active = True
        initialize_voice()

def say_chunk(text: str):
    """Speak a chunk of text immediately"""
    if not text.strip():
        return
    
    # For real-time streaming, speak immediately
    speak(text)

def finish_speech():
    """Finish speech streaming"""
    global streaming_active
    with streaming_lock:
        streaming_active = False
    
    # Wait for any ongoing speech to complete
    time.sleep(0.1)

# Utility functions
def speak_to_file(text: str, filename: str = "debug_output.wav") -> bool:
    """Save synthesized speech to WAV file for debugging"""
    if not text.strip():
        return False
    
    initialize_voice()
    if not voice:
        return False
    
    try:
        audio_data, sample_rate = _synthesize_audio(text)
        if audio_data is None:
            return False
        
        # Save to file
        sf.write(filename, audio_data, sample_rate)
        print(f"[TTS DEBUG] Saved audio to {filename}")
        return True
        
    except Exception as e:
        print(f"[TTS ERROR] Failed to save to file: {e}")
        return False

def test_tts():
    """Test TTS functionality"""
    test_text = "Text to speech system is working correctly."
    print("[TTS TEST] Testing speech synthesis...")
    
    if speak(test_text):
        print("[TTS TEST] ✅ TTS test successful")
        return True
    else:
        print("[TTS TEST] ❌ TTS test failed")
        return False

# Cleanup function
def cleanup_tts():
    """Cleanup TTS resources"""
    stop_tts_worker()
    # Clear any remaining audio
    try:
        sd.stop()
    except:
        pass

# Initialize on import if voice files exist
if __name__ == "__main__":
    # Test the TTS system
    test_tts()
else:
    # Check if voice files exist
    voice_path = "voices/en_US-kathleen-low.onnx"
    if os.path.exists(voice_path):
        initialize_voice()
    else:
        print(f"[TTS WARNING] Voice model not found at {voice_path}")