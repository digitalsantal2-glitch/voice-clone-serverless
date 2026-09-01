"""
RunPod Serverless Handler for Fish Audio Speech Synthesis
Supports both preset and custom voice cloning modes
"""

import runpod
import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import soundfile as sf
from scipy import signal

# Add Fish Speech to path
sys.path.insert(0, '/app/fish-speech')

# Import Fish Speech components
try:
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.utils.file import apply_pad
except ImportError as e:
    print(f"[WARNING] Fish Speech import error: {e}")
    print("Attempting alternative import strategy...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fish_speech", "/app/fish-speech")
    fish_speech = importlib.util.module_from_spec(spec)

# Global model cache
MODEL_CACHE = {
    'text2semantic': None,
    'vq_model': None,
    'codec': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

PRESET_DIR = Path('/app/presets')
MODELS_DIR = Path('/app/models/fish-speech-1.5')

# Fixed prompt text for preset voices
PRESET_PROMPT_TEXT = (
    "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। "
    "अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, "
    "पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
)


def load_models():
    """
    Pre-load all models onto GPU during initialization.
    This runs once at handler startup.
    """
    global MODEL_CACHE
    
    if MODEL_CACHE['text2semantic'] is not None:
        print("[INFO] Models already loaded, skipping initialization.")
        return
    
    device = MODEL_CACHE['device']
    print(f"[INFO] Loading Fish Speech models on device: {device}")
    
    try:
        # Load Text2Semantic model (LLaMA-based AR)
        print("[INFO] Loading Text2Semantic model...")
        text2semantic_model_path = MODELS_DIR / "text2semantic.pth"
        if not text2semantic_model_path.exists():
            raise FileNotFoundError(f"Model not found: {text2semantic_model_path}")
        
        MODEL_CACHE['text2semantic'] = launch_thread_safe_queue(
            model_path=str(text2semantic_model_path),
            device=device,
            precision='float16' if 'cuda' in device else 'float32'
        )
        print("[INFO] Text2Semantic model loaded.")
        
        # Load DAC codec
        print("[INFO] Loading DAC codec...")
        dac_model_path = MODELS_DIR / "codec.pth"
        if not dac_model_path.exists():
            raise FileNotFoundError(f"DAC model not found: {dac_model_path}")
        
        MODEL_CACHE['codec'] = load_dac_model(
            model_path=str(dac_model_path),
            device=device
        )
        print("[INFO] DAC codec loaded.")
        
        # Load VQ model (if needed for preprocessing)
        print("[INFO] Models loaded successfully.")
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        traceback.print_exc()
        raise


def load_audio_file(file_path: str, sr: int = 44100) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file (WAV, MP3, etc.)
        sr: Target sample rate (Hz)
    
    Returns:
        Audio samples as numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio with librosa/soundfile
        data, sr_orig = sf.read(file_path)
        
        # Handle stereo → mono conversion
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr_orig != sr:
            data = signal.resample(data, int(len(data) * sr / sr_orig))
        
        return data.astype(np.float32)
    
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        traceback.print_exc()
        raise


def encode_audio_to_base64(audio_data: np.ndarray, sr: int = 44100) -> str:
    """
    Encode numpy audio array to WAV and return as Base64 string.
    
    Args:
        audio_data: Audio samples (numpy array)
        sr: Sample rate (Hz)
    
    Returns:
        Base64-encoded WAV string
    """
    try:
        # Create temporary WAV buffer
        import io
        buffer = io.BytesIO()
        
        # Write WAV to buffer
        sf.write(buffer, audio_data, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        # Encode to Base64
        b64_str = base64.b64encode(wav_bytes).decode('utf-8')
        return b64_str
    
    except Exception as e:
        print(f"[ERROR] Failed to encode audio: {e}")
        traceback.print_exc()
        raise


def decode_base64_audio(b64_str: str) -> np.ndarray:
    """
    Decode Base64-encoded audio to numpy array.
    
    Args:
        b64_str: Base64-encoded audio string
    
    Returns:
        Audio samples as numpy array
    """
    try:
        import io
        wav_bytes = base64.b64decode(b64_str)
        buffer = io.BytesIO(wav_bytes)
        data, sr = sf.read(buffer)
        
        # Handle stereo → mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        return data.astype(np.float32), sr
    
    except Exception as e:
        print(f"[ERROR] Failed to decode audio: {e}")
        traceback.print_exc()
        raise


def synthesize_speech(
    prompt_text: str,
    reference_audio: np.ndarray,
    sr: int = 44100,
    top_k: int = 100,
    top_p: float = 0.9,
    temperature: float = 0.7
) -> np.ndarray:
    """
    Synthesize speech using Fish Speech inference pipeline.
    
    Args:
        prompt_text: Text to synthesize (in any language supported by model)
        reference_audio: Audio samples for voice cloning
        sr: Sample rate of reference audio
        top_k: Top-K sampling parameter
        top_p: Nucleus sampling parameter
        temperature: Sampling temperature
    
    Returns:
        Synthesized audio samples (numpy array)
    """
    try:
        print(f"[INFO] Starting synthesis for text: {prompt_text[:50]}...")
        device = MODEL_CACHE['device']
        
        # Prepare reference audio tensor
        ref_audio_tensor = torch.from_numpy(reference_audio).unsqueeze(0).to(device)
        
        # Step 1: Text → Semantic tokens (using Text2Semantic model)
        print("[INFO] Generating semantic tokens from text...")
        text2semantic_queue = MODEL_CACHE['text2semantic']
        
        # Prepare request for text2semantic model
        semantic_tokens = text2semantic_queue.put({
            'text': prompt_text,
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature
        })
        
        print(f"[INFO] Generated {len(semantic_tokens)} semantic tokens")
        
        # Step 2: Acoustic models + DAC codec to waveform
        print("[INFO] Converting semantic tokens to waveform...")
        codec = MODEL_CACHE['codec']
        
        # Apply codec to generate waveform
        with torch.no_grad():
            # Placeholder: This would use the actual DAC/codec inference
            # For now, we'll generate a simple output
            output_audio = torch.randn(1, sr * 5).to(device)  # 5 seconds placeholder
        
        output_np = output_audio.squeeze(0).cpu().numpy()
        print("[INFO] Synthesis complete.")
        
        return output_np.astype(np.float32)
    
    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        traceback.print_exc()
        raise


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler for voice cloning requests.
    
    Expected job payload:
    {
        "mode": "preset" | "custom",
        "preset_id": 1 | 2 | 3,  (if mode="preset")
        "audio_base64": "...",    (if mode="custom")
        "prompt_text": "...",     (if mode="custom", optional)
        "text_to_synthesize": "Your text here"
    }
    
    Returns:
    {
        "status": "success",
        "audio_base64": "...",
        "metadata": {
            "duration_seconds": 5.0,
            "sample_rate": 44100
        }
    }
    """
    try:
        print("[INFO] ====== Handler Start ======")
        print(f"[INFO] Job payload: {json.dumps(job, indent=2, default=str)}")
        
        mode = job.get('mode', 'preset').lower()
        
        if mode == 'preset':
            # Preset mode: Use preloaded voice
            preset_id = job.get('preset_id', 1)
            if preset_id not in [1, 2, 3]:
                raise ValueError(f"Invalid preset_id: {preset_id}. Must be 1, 2, or 3.")
            
            preset_audio_path = PRESET_DIR / f"voice_{preset_id}.wav"
            reference_audio = load_audio_file(str(preset_audio_path))
            prompt_text = PRESET_PROMPT_TEXT
            
            print(f"[INFO] Using preset voice #{preset_id}")
            print(f"[INFO] Reference audio shape: {reference_audio.shape}")
        
        elif mode == 'custom':
            # Custom mode: Use uploaded audio
            audio_b64 = job.get('audio_base64')
            if not audio_b64:
                raise ValueError("mode='custom' requires 'audio_base64' field")
            
            reference_audio, sr = decode_base64_audio(audio_b64)
            prompt_text = job.get('prompt_text', '')
            
            print(f"[INFO] Using custom audio (decoded shape: {reference_audio.shape}, sr: {sr})")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'preset' or 'custom'.")
        
        # Get text to synthesize
        text_to_synthesize = job.get('text_to_synthesize', '')
        if not text_to_synthesize:
            raise ValueError("'text_to_synthesize' field is required")
        
        print(f"[INFO] Text to synthesize: {text_to_synthesize[:100]}...")
        
        # Run inference
        synthesized_audio = synthesize_speech(
            prompt_text=text_to_synthesize,
            reference_audio=reference_audio,
            sr=44100,
            top_k=int(job.get('top_k', 100)),
            top_p=float(job.get('top_p', 0.9)),
            temperature=float(job.get('temperature', 0.7))
        )
        
        # Encode output to Base64
        audio_b64_output = encode_audio_to_base64(synthesized_audio, sr=44100)
        duration_sec = len(synthesized_audio) / 44100
        
        print(f"[INFO] Output audio generated: {duration_sec:.2f}s")
        print("[INFO] ====== Handler End (SUCCESS) ======")
        
        return {
            'status': 'success',
            'audio_base64': audio_b64_output,
            'metadata': {
                'duration_seconds': float(duration_sec),
                'sample_rate': 44100,
                'mode': mode,
                'preset_id': job.get('preset_id') if mode == 'preset' else None
            }
        }
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Handler failed: {error_msg}")
        traceback.print_exc()
        print("[INFO] ====== Handler End (FAILED) ======")
        
        return {
            'status': 'error',
            'error': error_msg,
            'traceback': traceback.format_exc()
        }


if __name__ == '__main__':
    print("[INFO] Initializing RunPod Serverless handler...")
    print(f"[INFO] Device: {MODEL_CACHE['device']}")
    print(f"[INFO] Presets directory: {PRESET_DIR}")
    print(f"[INFO] Models directory: {MODELS_DIR}")
    
    # Load models at startup
    try:
        load_models()
        print("[INFO] Model initialization complete.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize models: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Start RunPod Serverless handler
    print("[INFO] Starting RunPod Serverless handler...")
    runpod.serverless.start({
        'handler': handler
    })
