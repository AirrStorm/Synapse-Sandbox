#!/usr/bin/env python3
"""
Real-Time Speech-to-Text using Faster Whisper
Handles continuous audio capture and transcription
"""

import sounddevice as sd
import numpy as np
import queue
import time
import threading
from faster_whisper import WhisperModel
from threading import Event, Lock
import collections

# Configuration
VOLUME_THRESHOLD_DB = -25  # Base threshold in dB
SILENCE_TIMEOUT = 1.5  # seconds - time to wait after speech stops
MIN_AUDIO_LENGTH = 0.3  # minimum seconds of audio before transcription
MAX_AUDIO_LENGTH = 30  # maximum seconds to prevent memory issues
SAMPLE_RATE = 16000  # Whisper's native sample rate
DEVICE_ID = None  # None = default device

# Global state
audio_queue = queue.Queue(maxsize=100)  # Prevent memory buildup
mic_paused_event = Event()
model_lock = Lock()  # Prevent concurrent model access
model = None
transcription_callback = None

# Audio statistics for dynamic threshold adjustment
recent_volumes = collections.deque(maxlen=50)
background_noise_level = -40  # Initial estimate
audio_buffer = np.array([], dtype=np.float32)
last_speech_time = time.time()
processing_thread = None

# Thread safety
buffer_lock = Lock()

def initialize_model():
    """Initialize Whisper model once to avoid repeated loading"""
    global model
    if model is None:
        with model_lock:
            if model is None:  # Double-check pattern
                try:
                    print("[WHISPER] Loading model...")
                    model = WhisperModel(
                        "distil-large-v3", 
                        device="cpu", 
                        compute_type="int8",
                        num_workers=1  # Prevent threading issues
                    )
                    print("[WHISPER] Model loaded successfully")
                except Exception as e:
                    print(f"[WHISPER ERROR] Failed to load model: {e}")
                    model = None

def calculate_volume_db(data):
    """Calculate volume in dB with better numerical stability"""
    if len(data) == 0:
        return -60  # Very quiet
    
    # Use RMS for better volume estimation
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 1e-8:  # Prevent log(0)
        return -60
    
    volume_db = 20 * np.log10(rms)
    return volume_db

def update_background_noise(volume_db):
    """Dynamically adjust background noise estimation"""
    global background_noise_level
    recent_volumes.append(volume_db)
    
    if len(recent_volumes) >= 20:
        # Use 25th percentile as background noise estimate
        background_noise_level = np.percentile(recent_volumes, 25)

def is_speech_detected(data):
    """Improved speech detection with dynamic thresholding"""
    volume_db = calculate_volume_db(data)
    update_background_noise(volume_db)
    
    # Dynamic threshold: background noise + margin
    dynamic_threshold = max(background_noise_level + 10, VOLUME_THRESHOLD_DB)
    
    return volume_db > dynamic_threshold

def audio_callback(indata, frames, time_info, status):
    """Audio callback with better error handling"""
    if status:
        print(f"[AUDIO WARNING] {status}")
    
    # Skip if paused
    if mic_paused_event.is_set():
        return
    
    try:
        # Put audio data in queue (non-blocking to prevent callback delays)
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        # Drop oldest audio if queue is full
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(indata.copy())
        except queue.Empty:
            pass

def transcribe_audio(audio_data):
    """Transcribe audio with error handling and optimization"""
    if model is None:
        print("[WHISPER ERROR] Model not initialized")
        return ""
    
    try:
        with model_lock:
            # Ensure audio is the right length and format
            if len(audio_data) < MIN_AUDIO_LENGTH * SAMPLE_RATE:
                return ""  # Too short to be meaningful
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Transcribe with optimized settings
            segments, info = model.transcribe(
                audio_data,
                beam_size=1,  # Faster than 5, still good quality
                language="en",  # Specify language for speed
                condition_on_previous_text=False,  # Prevent error propagation
                temperature=0.0,  # Deterministic output
                vad_filter=True,  # Built-in voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine segments and clean up
            transcript = " ".join([segment.text.strip() for segment in segments])
            transcript = transcript.strip()
            
            # Filter out common Whisper hallucinations
            hallucination_phrases = [
                "thank you", "thanks for watching", "subscribe", 
                "you", ".", "...", "", "music", "applause"
            ]
            
            if (transcript.lower() in hallucination_phrases or 
                len(transcript) < 3 or
                transcript.count('.') > len(transcript.split()) // 2):  # Too many dots
                return ""
                
            return transcript
            
    except Exception as e:
        print(f"[TRANSCRIPTION ERROR] {e}")
        return ""

def process_transcription_async(audio_data, callback):
    """Process transcription in separate thread to avoid blocking"""
    transcript = transcribe_audio(audio_data)
    if transcript and callback:
        print(f"[TRANSCRIPT] {transcript}")
        try:
            callback(transcript)
        except Exception as e:
            print(f"[CALLBACK ERROR] {e}")

def audio_processing_loop():
    """Main audio processing loop - handles buffering and transcription triggers"""
    global audio_buffer, last_speech_time, processing_thread
    
    while not mic_paused_event.is_set():
        try:
            # Get audio with timeout
            audio_chunk = audio_queue.get(timeout=0.1)
            
            with buffer_lock:
                # Check if this chunk contains speech
                if is_speech_detected(audio_chunk):
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk.flatten()])
                    last_speech_time = time.time()
                    
                    # Prevent buffer from getting too long
                    max_samples = MAX_AUDIO_LENGTH * SAMPLE_RATE
                    if len(audio_buffer) > max_samples:
                        # Keep only the most recent audio
                        audio_buffer = audio_buffer[-max_samples:]
                
                # Check for silence timeout or buffer overflow
                silence_duration = time.time() - last_speech_time
                buffer_duration = len(audio_buffer) / SAMPLE_RATE
                
                should_transcribe = (
                    len(audio_buffer) > 0 and (
                        silence_duration > SILENCE_TIMEOUT or
                        buffer_duration > MAX_AUDIO_LENGTH
                    )
                )
                
                if should_transcribe and buffer_duration >= MIN_AUDIO_LENGTH:
                    # Wait for previous transcription to complete
                    if processing_thread and processing_thread.is_alive():
                        processing_thread.join(timeout=0.5)
                    
                    # Start new transcription in background
                    audio_to_process = audio_buffer.copy()
                    audio_buffer = np.array([], dtype=np.float32)
                    
                    if transcription_callback:
                        processing_thread = threading.Thread(
                            target=process_transcription_async,
                            args=(audio_to_process, transcription_callback),
                            daemon=True
                        )
                        processing_thread.start()
        
        except queue.Empty:
            # Check if we have a long buffer that should be processed
            with buffer_lock:
                buffer_duration = len(audio_buffer) / SAMPLE_RATE
                if buffer_duration >= MIN_AUDIO_LENGTH * 2:  # Process longer buffers
                    if processing_thread is None or not processing_thread.is_alive():
                        audio_to_process = audio_buffer.copy()
                        audio_buffer = np.array([], dtype=np.float32)
                        
                        if transcription_callback:
                            processing_thread = threading.Thread(
                                target=process_transcription_async,
                                args=(audio_to_process, transcription_callback),
                                daemon=True
                            )
                            processing_thread.start()
            continue
        except Exception as e:
            print(f"[PROCESSING ERROR] {e}")

def start_transcription(callback, device_id=DEVICE_ID, sample_rate=SAMPLE_RATE):
    """Main transcription loop with improved buffer management"""
    global transcription_callback, audio_buffer, last_speech_time
    
    # Initialize
    initialize_model()
    if model is None:
        print("[ERROR] Cannot start transcription - Whisper model failed to load")
        return
    
    transcription_callback = callback
    
    # Reset state
    with buffer_lock:
        audio_buffer = np.array([], dtype=np.float32)
        last_speech_time = time.time()
    
    print(f"[TRANSCRIPTION] Starting with device {device_id} at {sample_rate}Hz...")
    
    # Start audio processing thread
    processing_loop_thread = threading.Thread(target=audio_processing_loop, daemon=True)
    processing_loop_thread.start()
    
    try:
        with sd.InputStream(
            samplerate=sample_rate, 
            device=device_id, 
            channels=1,
            dtype='float32', 
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1)  # 100ms blocks for responsiveness
        ):
            print("[TRANSCRIPTION] Audio stream started. Listening...")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
                # Check if processing thread died
                if not processing_loop_thread.is_alive():
                    print("[WARNING] Processing thread died, restarting...")
                    processing_loop_thread = threading.Thread(target=audio_processing_loop, daemon=True)
                    processing_loop_thread.start()
                    
    except KeyboardInterrupt:
        print("\n[TRANSCRIPTION] Stopping...")
    except Exception as e:
        print(f"[TRANSCRIPTION ERROR] {e}")
    finally:
        # Cleanup
        transcription_callback = None
        clear_audio_buffer()

def pause_microphone():
    """Pause microphone input"""
    mic_paused_event.set()
    print("[MIC] Paused")

def resume_microphone():
    """Resume microphone input"""
    mic_paused_event.clear()
    print("[MIC] Resumed")

def clear_audio_buffer():
    """Clear pending audio data"""
    global audio_buffer
    
    # Clear the queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    
    # Clear the buffer
    with buffer_lock:
        audio_buffer = np.array([], dtype=np.float32)

def get_audio_stats():
    """Get current audio statistics for debugging"""
    with buffer_lock:
        buffer_duration = len(audio_buffer) / SAMPLE_RATE
        queue_size = audio_queue.qsize()
        
    return {
        'buffer_duration': buffer_duration,
        'queue_size': queue_size,
        'background_noise_level': background_noise_level,
        'mic_paused': mic_paused_event.is_set(),
        'recent_volume_samples': len(recent_volumes)
    }

def test_transcription():
    """Test transcription functionality"""
    def test_callback(transcript):
        print(f"[TEST] Received: {transcript}")
    
    print("[TEST] Testing transcription for 10 seconds...")
    print("[TEST] Please speak something...")
    
    try:
        import threading
        import time
        
        # Start transcription in a thread
        transcription_thread = threading.Thread(
            target=start_transcription, 
            args=(test_callback,),
            daemon=True
        )
        transcription_thread.start()
        
        # Wait for 10 seconds
        time.sleep(10)
        
        print("[TEST] Test completed")
        return True
        
    except Exception as e:
        print(f"[TEST ERROR] {e}")
        return False

if __name__ == "__main__":
    # Test the transcription system
    test_transcription()