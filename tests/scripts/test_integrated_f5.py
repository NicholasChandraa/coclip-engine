import os
import sys

# Add engine directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.utils.tts_engine import TTSEngine
from app.utils.logging import logger

def test_integration():
    engine = TTSEngine()
    
    output_path = "tests/output/integrated_f5_test.wav"
    text = "Halo, ini adalah pengujian integrasi F5-TTS menggunakan suara reporter cewek. Sekarang sistem sudah otomatis unload GPU setelah selesai."
    
    print(f"Starting integrated synthesis...")
    try:
        path = engine.synthesize(text, "id", output_path)
        if os.path.exists(path):
            print(f"Success! Generated at: {path}")
        else:
            print("Failed: Output file not created.")
    except Exception as e:
        print(f"Error during integrated test: {e}")
    finally:
        engine.unload()


if __name__ == "__main__":
    test_integration()
