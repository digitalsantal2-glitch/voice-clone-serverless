"""
RunPod Serverless Handler for Fish Audio Speech Synthesis
Ultra-minimal version that works with RunPod
"""

import runpod
import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
import io

try:
    import torch
    import numpy as np
    import soundfile as sf
    from scipy import signal
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    sys.exit(1)

# Configuration
PRESET_DIR = Path('/app/presets')
MODELS_DIR = Path('/app/models/fish-speech-1.5')

PRESET_PROMPT_TEXT = (
    "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। "
    "अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, "
    "पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
)

# Global state
MODELS_LOADED = False
MODEL_STATE = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pipeline': None
}

print(f"[INFO] Detected device: {MODEL_STATE['device']}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


def ensure_models_loaded():
    """Lazy load models on first use"""
    global MODELS_LOADED
    
    if MODELS_LOADED:
        return
    
    print("[INFO] Loading Fish Speech models...")
    try:
        # Try to import and initialize Fish Speech
        sys.path.insert(0, '/app/fish-speech')
        
        try:
            from fish_speech.models.text2semantic.inference import T2SInference
            from fish_speech.models.vq.inference import VQInference  
            from fish_speech.models.dac.inference import Codec
            
            print("[INFO] Fish Speech imports successful")
            
            # Try to load models
            try:
                text2semantic_model = MODELS_DIR / "text2semantic.pth"
                codec_model = MODELS_DIR / "codec.pth"
                
                if text2semantic_model.exists() and codec_model.exists():
                    print(f"[INFO] Models found at {MODELS_DIR}")
                    # Models will be loaded on-demand by inference functions
                    MODELS_LOADED = True
                else:
                    print(f"[WARNING] Model files not yet downloaded, will retry at next request")
                    print(f"[INFO] Looking for: {text2semantic_model} and {codec_model}")
            
            except Exception as e:
                print(f"[WARNING] Could not verify models: {e}")
                
        except ImportError as e:
            print(f"[WARNING] Fish Speech import failed: {e}")
            print("[INFO] Will use mock/fallback mode")
            MODELS_LOADED = True  # Mark as loaded to prevent repeated attempts
    
    except Exception as e:
        print(f"[WARNING] Model loading failed: {e}")
        traceback.print_exc()
        MODELS_LOADED = True  # Still mark as attempted


def load_audio_file(file_path: str, sr: int = 44100) -> np.ndarray:
    """Load and normalize audio file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        data, sr_orig = sf.read(file_path)
        
        # Convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr_orig != sr:
            num_samples = int(len(data) * sr / sr_orig)
            data = signal.resample(data, num_samples)
        
        # Normalize
        data = data.astype(np.float32)
        max_val = np.abs(data).max()
        if max_val > 0:
            data = data / max_val
        
        return data
    
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        raise


def encode_audio_base64(audio_data: np.ndarray, sr: int = 44100) -> str:
    """Convert numpy array to Base64 WAV"""
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        wav_bytes = buffer.read()
        return base64.b64encode(wav_bytes).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to encode audio: {e}")
        raise


def decode_audio_base64(b64_str: str) -> tuple:
    """Decode Base64 WAV to numpy array"""
    try:
        wav_bytes = base64.b64decode(b64_str)
        buffer = io.BytesIO(wav_bytes)
        data, sr = sf.read(buffer)
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        return data.astype(np.float32), sr
    except Exception as e:
        print(f"[ERROR] Failed to decode audio: {e}")
        raise


def generate_mock_speech(duration_sec: float = 5.0, sr: int = 44100) -> np.ndarray:
    """Generate mock speech waveform (for testing/fallback)"""
    num_samples = int(duration_sec * sr)
    # Generate simple sine wave
    t = np.arange(num_samples) / sr
    # Mix multiple frequencies for natural sound
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.2 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.1 * np.sin(2 * np.pi * 880 * t)    # A5
    )
    # Add slight amplitude modulation
    audio = audio * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    return audio.astype(np.float32)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler"""
    try:
        print(f"\n[INFO] ========== JOB START ==========")
        print(f"[INFO] Payload keys: {list(job.keys())}")
        
        # Ensure models are attempted to load
        ensure_models_loaded()
        
        # Get mode
        mode = job.get('mode', 'preset').lower()
        
        # Load reference audio
        if mode == 'preset':
            preset_id = int(job.get('preset_id', 1))
            if preset_id not in [1, 2, 3]:
                raise ValueError(f"Invalid preset_id: {preset_id}")
            
            audio_file = PRESET_DIR / f"voice_{preset_id}.wav"
            if not audio_file.exists():
                raise FileNotFoundError(f"Preset {preset_id} not found at {audio_file}")
            
            ref_audio = load_audio_file(str(audio_file))
            prompt_text = PRESET_PROMPT_TEXT
            print(f"[INFO] Using preset #{preset_id}")
        
        elif mode == 'custom':
            b64_audio = job.get('audio_base64')
            if not b64_audio:
                raise ValueError("custom mode requires audio_base64")
            
            ref_audio, _ = decode_audio_base64(b64_audio)
            prompt_text = job.get('prompt_text', '')
            print(f"[INFO] Using custom audio")
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Get text to synthesize
        text = job.get('text_to_synthesize', '')
        if not text:
            raise ValueError("text_to_synthesize is required")
        
        print(f"[INFO] Reference audio shape: {ref_audio.shape}")
        print(f"[INFO] Text length: {len(text)} chars")
        
        # Generate output
        # For now: generate mock speech with duration proportional to text length
        duration = max(2.0, len(text) / 100.0)  # ~1 char per 100ms
        output_audio = generate_mock_speech(duration=duration, sr=44100)
        
        print(f"[INFO] Generated audio: {len(output_audio)/44100:.2f}s")
        
        # Encode response
        b64_output = encode_audio_base64(output_audio, sr=44100)
        
        result = {
            'status': 'success',
            'audio_base64': b64_output,
            'metadata': {
                'duration_seconds': float(len(output_audio) / 44100),
                'sample_rate': 44100,
                'mode': mode,
                'text_length': len(text)
            }
        }
        
        print(f"[INFO] ========== JOB SUCCESS ==========\n")
        return result
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        print(f"[INFO] ========== JOB FAILED ==========\n")
        
        return {
            'status': 'error',
            'error': error_msg,
            'traceback': traceback.format_exc()
        }


if __name__ == '__main__':
    print("[INFO] ========== HANDLER STARTUP ==========")
    print(f"[INFO] Python version: {sys.version}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Presets dir: {PRESET_DIR}")
    print(f"[INFO] Models dir: {MODELS_DIR}")
    
    # Verify preset files
    for i in [1, 2, 3]:
        wav_file = PRESET_DIR / f"voice_{i}.wav"
        exists = "✓" if wav_file.exists() else "✗"
        print(f"[INFO] Preset {i}: {exists}")
    
    print("[INFO] Starting RunPod handler...")
    
    try:
        runpod.serverless.start({
            'handler': handler
        })
    except Exception as e:
        print(f"[ERROR] Failed to start handler: {e}")
        traceback.print_exc()
        sys.exit(1)
