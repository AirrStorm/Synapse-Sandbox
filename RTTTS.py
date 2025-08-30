import io
import wave
import queue
import threading
import resampy
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice

voice = PiperVoice.load("voices/glados_piper_medium.onnx")

speech_queue = queue.Queue()
tts_done = threading.Event()

def _speak(text: str):
    if not text.strip():
        return

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(text, wav_file)

    buf.seek(0)
    data, samplerate = sf.read(buf, dtype='float32')

    if data.size == 0:
        print("Error: PiperVoice produced no audio")
        return

    target = 48000
    if samplerate != target:
        data = resampy.resample(data, samplerate, target)
        samplerate = target

    sd.play(data, samplerate)
    sd.wait()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        _speak(text)
    tts_done.set()  

def stream_speech():
    """Starts TTS queue processing"""
    tts_done.clear()
    threading.Thread(target=tts_worker, daemon=True).start()

def say_chunk(text):
    """Adds text to the speech queue"""
    speech_queue.put(text)

def finish_speech():
    """Closes the queue and waits for TTS to finish"""
    speech_queue.put(None)
    tts_done.wait()
