#!/usr/bin/env python3
"""
Real-time Speech-to-Text with AI Integration
Main entry point for the application
"""

import sys
import signal
import time
import atexit
from RTSTT import start_transcription, pause_microphone, resume_microphone, mic_paused_event, initialize_model, get_audio_stats
from ollama_client import send_to_ollama, test_ollama_connection, shutdown_ollama_client
from RTTTS import test_tts, cleanup_tts

# Global state for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle graceful shutdown on Ctrl+C"""
    global shutdown_requested
    if shutdown_requested:
        print("\n🛑 Force quitting...")
        sys.exit(1)
    
    shutdown_requested = True
    print("\n🛑 Shutting down gracefully...")
    print("   (Press Ctrl+C again to force quit)")
    
    # Initiate cleanup
    cleanup_application()
    sys.exit(0)

def cleanup_application():
    """Comprehensive cleanup of all components"""
    print("🧹 Cleaning up...")
    
    try:
        # Stop microphone first
        pause_microphone()
        print("  ✅ Microphone stopped")
    except Exception as e:
        print(f"  ❌ Microphone cleanup error: {e}")
    
    try:
        # Cleanup AI client
        shutdown_ollama_client()
        print("  ✅ AI client shutdown")
    except Exception as e:
        print(f"  ❌ AI client cleanup error: {e}")
    
    try:
        # Cleanup TTS
        cleanup_tts()
        print("  ✅ TTS cleanup")
    except Exception as e:
        print(f"  ❌ TTS cleanup error: {e}")
    
    print("🔇 All systems stopped")

def check_system_requirements():
    """Check if all required components are available"""
    print("🔍 Checking system requirements...")
    
    # Check imports
    try:
        import sounddevice
        import numpy
        import faster_whisper
        import ollama
        print("  ✅ Required Python packages installed")
    except ImportError as e:
        print(f"  ❌ Missing required package: {e}")
        print("Install missing packages with:")
        print("  pip install faster-whisper sounddevice numpy ollama-python")
        return False
    
    # Check Piper voice files
    import os
    voice_path = "voices/en_US-kathleen-low.onnx"
    if not os.path.exists(voice_path):
        print(f"  ⚠️  Voice model not found at {voice_path}")
        print("  TTS may not work properly")
    else:
        print("  ✅ TTS voice model found")
    
    return True

def run_system_tests():
    """Run comprehensive system tests"""
    print("🧪 Running system tests...")
    
    # Test Whisper model loading
    print("  Testing speech recognition...")
    try:
        initialize_model()
        print("  ✅ Speech recognition model loaded")
    except Exception as e:
        print(f"  ❌ Speech recognition test failed: {e}")
        return False
    
    # Test TTS
    print("  Testing text-to-speech...")
    if test_tts():
        print("  ✅ Text-to-speech working")
    else:
        print("  ⚠️  Text-to-speech test failed (may still work)")
    
    # Test Ollama connection
    print("  Testing AI connection...")
    if test_ollama_connection():
        print("  ✅ AI connection successful")
    else:
        print("  ❌ AI connection failed")
        print("  Make sure Ollama is running: ollama serve")
        print("  And model is installed: ollama pull mistral")
        return False
    
    return True

def print_usage_instructions():
    """Print detailed usage instructions"""
    print("=" * 60)
    print("🎙️  REAL-TIME SPEECH-TO-TEXT WITH AI INTEGRATION")
    print("=" * 60)
    print()
    print("📋 CONTROLS:")
    print("  • Speak naturally to interact with the AI")
    print("  • The system will automatically detect when you start/stop speaking")
    print("  • Press Ctrl+C to stop the application")
    print()
    print("🗣️  VOICE COMMANDS:")
    print("  • 'Clear history' - Start a new conversation")
    print("  • 'Show history' - Hear conversation summary")
    print("  • 'Stop talking' - Make the AI be quiet")
    print()
    print("⚙️  SYSTEM STATUS:")
    print(f"  • Microphone: {'🟢 Active' if not mic_paused_event.is_set() else '🔴 Paused'}")
    
    # Show audio stats
    try:
        stats = get_audio_stats()
        print(f"  • Audio buffer: {stats['buffer_duration']:.2f}s")
        print(f"  • Background noise: {stats['background_noise_level']:.1f}dB")
    except:
        print("  • Audio stats: Not available")
    
    print()
    print("=" * 60)

def main():
    """Main application entry point"""
    global shutdown_requested
    
    print("🚀 Starting Real-time Speech-to-Text with AI Integration")
    
    # Set up graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_application)
    
    try:
        # Check system requirements
        if not check_system_requirements():
            print("❌ System requirements not met. Exiting.")
            return 1
        
        # Run system tests
        if not run_system_tests():
            print("❌ System tests failed. Please check your configuration.")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: ollama serve")
            print("2. Install the AI model: ollama pull mistral")
            print("3. Check your microphone permissions")
            return 1
        
        # Ensure microphone starts unpaused
        if mic_paused_event.is_set():
            resume_microphone()
        
        # Print usage instructions
        print_usage_instructions()
        
        print("🎤 Microphone active - listening for speech...")
        print("📝 Transcription started. Speak now!")
        print()
        
        # Start the main transcription loop
        # This will block until interrupted
        start_transcription(send_to_ollama)
        
    except KeyboardInterrupt:
        # This should be handled by signal_handler, but keep as backup
        if not shutdown_requested:
            print("\n🛑 Received interrupt signal")
            cleanup_application()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure all required modules are installed:")
        print("  pip install faster-whisper sounddevice numpy ollama-python piper-tts")
        return 1
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check your configuration and try again.")
        
        # Try to provide helpful debugging info
        try:
            stats = get_audio_stats()
            print(f"Debug info: {stats}")
        except:
            pass
        
        return 1
        
    finally:
        # Final cleanup
        if not shutdown_requested:
            cleanup_application()
        print("👋 Goodbye!")
        
    return 0

def debug_mode():
    """Run in debug mode with extra logging"""
    print("🐛 DEBUG MODE ENABLED")
    print("This will show detailed system information and run diagnostics")
    print()
    
    # Enhanced system check
    check_system_requirements()
    
    # Audio device info
    try:
        import sounddevice as sd
        print("🎵 AUDIO DEVICES:")
        print(sd.query_devices())
        print()
    except Exception as e:
        print(f"Could not query audio devices: {e}")
    
    # Run tests
    run_system_tests()
    
    # Show continuous stats
    print("Starting in debug mode (stats will be shown every 5 seconds)...")
    
    def debug_stats_loop():
        while not shutdown_requested:
            try:
                stats = get_audio_stats()
                print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {stats}")
                time.sleep(5)
            except:
                break
    
    import threading
    stats_thread = threading.Thread(target=debug_stats_loop, daemon=True)
    stats_thread.start()
    
    # Run main application
    return main()

if __name__ == "__main__":
    # Check for debug mode
    if len(sys.argv) > 1 and sys.argv[1] in ["--debug", "-d", "debug"]:
        exit_code = debug_mode()
    else:
        exit_code = main()
    
    sys.exit(exit_code)