import io
import os
import sys
import base64
import tempfile
from pathlib import Path
import soundfile as sf
import torch
import runpod

# Fish Speech Imports
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine import TTSInferenceEngine

print("1. Loading Real Fish Speech AI Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load AI Engine
q = launch_thread_safe_queue("/app/models/fish-speech-1.5", device=device, precision=precision, compile=False)
dec = load_decoder_model(config_name="modded_dac_vq", checkpoint_path="/app/models/fish-speech-1.5/codec.pth", device=device)
engine = TTSInferenceEngine(llama_queue=q, decoder_model=dec, precision=precision, compile=False)
print("2. Real AI Engine is 100% Ready!")

PRESET_DIR = Path('/app/presets')
COMMON_PROMPT = "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"

def handler(job):
    inp = job.get('input', {})
    
    # Text to synthesize
    text = inp.get('text', '') or inp.get('text_to_synthesize', '')
    mode = inp.get('mode', 'preset')
    preset_id = inp.get('preset_id', 1)
    prompt_text = inp.get('prompt_text', '')
    ref_b64 = inp.get('reference_audio', '') or inp.get('audio_base64', '')

    if not text:
        return {"error": "text is required"}

    try:
        audio_bytes = None
        clean_prompt = prompt_text

        # 1. Preset Mode (आपकी 3 DJ आवाज़ें)
        if mode == 'preset':
            audio_path = PRESET_DIR / f"voice_{preset_id}.wav"
            if not audio_path.exists():
                audio_path = PRESET_DIR / "voice_1.wav"
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            clean_prompt = COMMON_PROMPT

        # 2. Custom User Upload Mode
        elif mode == 'custom' and ref_b64:
            audio_bytes = base64.b64decode(ref_b64)

        if not audio_bytes:
            return {"error": "No reference audio provided"}

        # Real AI Synthesis
        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=audio_bytes, text=clean_prompt if clean_prompt else "")]
        )

        for result in engine.inference(req):
            if result.code == "final":
                sr, audio_data = result.audio
                buf = io.BytesIO()
                sf.write(buf, audio_data, sr, format='WAV')
                return {
                    "status": "success",
                    "audio_base64": base64.b64encode(buf.getvalue()).decode('utf-8'),
                    "duration_seconds": float(len(audio_data) / sr)
                }
            elif result.code == "error":
                return {"error": str(result.error)}

        return {"error": "Generation failed"}
    except Exception as e:
        print("[ERROR]", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
