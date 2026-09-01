import io
import os
import base64
import tempfile
import soundfile as sf
import runpod
from voxcpm import VoxCPM

print("1. Loading OpenBMB VoxCPM2 Model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("2. Running Quick Warmup so AI is pre-compiled...")
try:
    _ = model.generate(text="नमस्ते", cfg_value=2.0, inference_timesteps=2)
    print("3. Warmup Done! System is Ready for 3-Second Speech Generation!")
except Exception as e:
    print("Warmup notice:", e)

# Pre-saved Audio Paths & Exact Prompts
PRESETS = {
    "voice1": {
        "file": "/app/presets/long_kolhapuri.wav",
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    },
    "voice2": {
        "file": "/app/presets/competition_dialogue.mp3",
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    },
    "voice3": {
        "file": "/app/presets/competition_voice.mp3",
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    }
}

def handler(job):
    inp = job.get('input', {})
    text = inp.get('text', '')                    # नया डायलॉग
    mode = inp.get('mode', 'preset')              # 'preset' या 'custom'
    preset_name = inp.get('preset_name', 'voice1')# 'voice1', 'voice2', 'voice3'
    prompt_text = inp.get('prompt_text', '')      # कस्टम प्रॉम्प्ट
    ref_audio_b64 = inp.get('reference_audio', '')# कस्टम ऑडियो

    if not text:
        return {"error": "Text is required"}

    try:
        ref_path = None
        active_prompt = None

        # 1. Preset Voice Mode (कंटेनर में पहले से मौजूद फाइलों से तुरंत पढ़ना)
        if mode == 'preset':
            preset_info = PRESETS.get(preset_name, PRESETS["voice1"])
            ref_path = preset_info["file"]
            active_prompt = preset_info["prompt"]

        # 2. Custom User Upload Mode
        elif mode == 'custom' and ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_path = tmp.name
            active_prompt = prompt_text if prompt_text else None

        if ref_path and os.path.exists(ref_path):
            wav = model.generate(
                text=text,
                prompt_wav_path=ref_path,
                prompt_text=active_prompt,
                cfg_value=2.0,
                inference_timesteps=10
            )
            if mode == 'custom':
                os.unlink(ref_path)
        else:
            wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)

        # 48kHz WAV Output Base64
        buf = io.BytesIO()
        sr = getattr(model.tts_model, "sample_rate", 48000)
        sf.write(buf, wav, sr, format='WAV')
        out_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "audio_base64": out_b64
        }
    except Exception as e:
        print("Inference Error:", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
