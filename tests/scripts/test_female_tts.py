"""
Test Indonesian Female TTS (Gadis Voice).
Model: jerichosiahaya/vits-tts-id
Gender: Female (Suara Cewe) - Speaker 81

Usage:
    python test_female_tts.py
"""

import os
import torch
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download

def normalize_indonesian(text: str) -> str:
    """
    Normalize standard Indonesian text to the model's specific IPA-ish vocabulary.
    """
    text = text.lower()
    
    # Specific word fixes (Phonetic hacks)
    word_fixes = {
        "oke": "okey",
        "okay": "okey",
        "halo": "halo",
        "video": "fidio",
    }
    
    for word, fix in word_fixes.items():
        # Use regex to match whole words only
        text = re.sub(r'\b' + re.escape(word) + r'\b', fix, text)

    # Mapping table
    # Standard -> Model Vocab (IPA-ish)
    mapping = {
        'ng': 'ŋ',
        'ny': 'ɲ',
        'sy': 'ʃ',
        'kh': 'x',
        'g': 'ɡ', # IPA g
        'j': 'dʒ', # Hard J (d + voiced postalveolar fricative)
        'c': 'tʃ', # Hard C (t + voiceless postalveolar fricative)
        'y': 'j',  # In IPA, 'j' is the sound of 'y'
        'v': 'f',
        'q': 'k',
    }
    
    # Sort keys by length descending to match 'ng' before 'n' or 'g'
    for k in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(k, mapping[k])
        
    return text

def test_female_synthesis(text: str, output_path: str):
    print(f"\n============================================================")
    print(f"Model: jerichosiahaya/vits-tts-id (Female - Gadis)")
    print(f"Text: {text}")
    print(f"============================================================")

    model_id = "jerichosiahaya/vits-tts-id"
    
    # Download model files manually to a dedicated directory
    print("Downloading/Locating model files in 'model-tts'...")
    # Use path relative to engine root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    model_dir = os.path.join(engine_root, "model-tts")
    os.makedirs(model_dir, exist_ok=True)
    
    config_path = hf_hub_download(repo_id=model_id, filename="config.json", local_dir=model_dir)
    checkpoint_path = hf_hub_download(repo_id=model_id, filename="checkpoint_1260000-inference.pth", local_dir=model_dir)
    speakers_path = hf_hub_download(repo_id=model_id, filename="speakers.pth", local_dir=model_dir)

    print(f"Loading model via TTS API (GPU={torch.cuda.is_available()})...")
    # Initialize TTS with the local files
    # We change CWD temporarily so the library finds speakers.pth next to the config
    old_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        tts = TTS(model_path=os.path.basename(checkpoint_path), 
                  config_path=os.path.basename(config_path), 
                  gpu=torch.cuda.is_available())
    finally:
        os.chdir(old_cwd)

    # Synthesize to file
    # Speaker 81 is 'gadis'
    print(f"Synthesizing with speaker 'gadis' (ID 81)...")
    
    # Normalize text for the model
    normalized_text = normalize_indonesian(text)
    print(f"Normalized Text: {normalized_text}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use speaker_id since it's a multi-speaker model
    tts.tts_to_file(text=normalized_text, speaker="gadis", file_path=output_path)
    
    print(f"\nResults:")
    print(f"  Output: {output_path}")

if __name__ == "__main__":
    sample_text = "Halo saya AIRA, AI Retail Assistant. Ada yang bisa aku bantu?"
    
    # Standard output location in engine/tests/test_output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_file = os.path.join(engine_root, "tests", "test_output", "tts", "female_id.wav")
    
    try:
        test_female_synthesis(sample_text, output_file)
        print("\nDone! Check output file in:", output_file)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
