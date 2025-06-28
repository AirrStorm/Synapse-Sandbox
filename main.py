from RTSTT import start_transcription, mic_paused_event
from ollama_client import send_to_ollama


if __name__ == "__main__":
    print("🎙️ Speak to interact (Ctrl+C to stop)...")
    try:
        # Ensure microphone starts unpaused
        mic_paused_event.clear()
        print(f"[INITIAL DEBUG] Microphone starts unpaused: {not mic_paused_event.is_set()}")

        start_transcription(send_to_ollama)

    except KeyboardInterrupt:
        print("\n🛑 Exiting application.")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
