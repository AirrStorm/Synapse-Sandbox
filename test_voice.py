import io
import wave
import resampy
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice

# Load your voice model
voice = PiperVoice.load(
    model_path="voices/glados_piper_medium.onnx",
    config_path="voices/glados_piper_medium.onnx.json"
)

def speak(text: str):
    if not text.strip():
        return

    print(f"\n{text}")

    buffer = io.BytesIO()

    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text, wav_file)

    buffer.seek(0)
    data, samplerate = sf.read(buffer, dtype='float32')

    # Resample to 48000 Hz (your device's default)
    target_samplerate = 48000
    if samplerate != target_samplerate:
        data = resampy.resample(data, samplerate, target_samplerate)
        samplerate = target_samplerate

    sd.play(data, samplerate)
    sd.wait()

# Test
speak("You're navigating these test chambers faster than I can build them")


