if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice

    def dummy_generator():
        yield "This is piper tts speaking."

    voice = PiperVoice(
        model_file="voices/glados_piper_medium.onnx",
        config_file="voices/glados_piper_medium.onnx.json",
    )

    engine = PiperEngine(
        piper_path="D:/Downloads/piper_windows_amd64/piper/piper.exe",
        voice=voice,
    )

    stream = TextToAudioStream(engine)
    stream.feed(dummy_generator())
    stream.play()