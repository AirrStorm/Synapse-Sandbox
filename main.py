from RTSTT import start_transcription, mic_paused_event
from ollama_client import send_to_ollama
from wakey import wait_for_wake_word  # <-- we'll create this

if __name__ == "__main__":
    print("🎙️ Waiting for 'Hey MIO' to start listening...")

    # Block until wake word detected
    if wait_for_wake_word(
        model_path="voices/Hey-MIO.ppn",
        mic_index=12,
        access_key="KcZFoUcTWB0a8mZaciQRWxwE3/LQM1hQAdc5XWLnl/NIDLtug1LbwA=="
    ):
        print("✅ Wake word detected — activating speech recognition...")

        try:
            mic_paused_event.clear()  # Ensure mic is unpaused
            start_transcription(send_to_ollama)

        except KeyboardInterrupt:
            print("\n🛑 Exiting application.")
        except Exception as e:
            print(f"\n💥 An unexpected error occurred: {e}")
