import tempfile
import resampy
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice
import wave

voice = PiperVoice.load("voices/glados_piper_medium.onnx")

def speak(text: str):
    if not text.strip():
        return

    print(f"\n{text}")

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        with wave.open(tmp_wav.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize_wav(text, wav_file)

        # Read WAV
        data, samplerate = sf.read(tmp_wav.name, dtype='float32')

    if len(data) == 0:
        print("Error: PiperVoice produced no audio")
        return

    # Resample if needed
    target_samplerate = 48000
    if samplerate != target_samplerate:
        data = resampy.resample(data, samplerate, target_samplerate)
        samplerate = target_samplerate

    # Play audio
    sd.play(data, samplerate)
    sd.wait()

# # Test
speak("You're navigating the test chambers faster than I can build them")

