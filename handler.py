import io
import os
import base64
import tempfile
import requests
import soundfile as sf
import runpod
from voxcpm import VoxCPM

print("Loading OpenBMB VoxCPM2 Model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("OpenBMB VoxCPM2 Ready!")

# ==============================================================================
# 🎯 तीनों आवाज़ों के अलग-अलग Prompt Texts (बाद में आप इन्हें यहाँ बदल सकते हैं)
# ==============================================================================
PRESETS = {
    "voice1": {
        "url": "https://files.catbox.moe/b1vfng.wav",
        "file": "/tmp/long_kolhapuri.wav",
        "name": "Long Kolhapuri",
        # 👉 Voice 1 का Prompt:
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    },
    "voice2": {
        "url": "https://files.catbox.moe/i87vs7.mp3",
        "file": "/tmp/competition_dialogue.mp3",
        "name": "Competition Dialogue",
        # 👉 Voice 2 का Prompt:
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    },
    "voice3": {
        "url": "https://files.catbox.moe/gr8o75.mp3",
        "file": "/tmp/competition_voice.mp3",
        "name": "Competition Voice",
        # 👉 Voice 3 का Prompt:
        "prompt": "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"
    }
}

def get_preset_audio(key):
    if key not in PRESETS:
        key = "voice1"
    info = PRESETS[key]
    if not os.path.exists(info["file"]):
        print(f"Downloading audio for {info['name']}...")
        r = requests.get(info["url"])
        with open(info["file"], "wb") as f:
            f.write(r.content)
    return info["file"], info["prompt"]

def handler(job):
    inp = job.get('input', {})
    text = inp.get('text', '')                    # नया डायलॉग
    mode = inp.get('mode', 'preset')              # 'preset' या 'custom'
    preset_name = inp.get('preset_name', 'voice1')# 'voice1', 'voice2', 'voice3'
    prompt_text = inp.get('prompt_text', '')      # कस्टम आवाज का प्रॉम्प्ट
    ref_audio_b64 = inp.get('reference_audio', '')# कस्टम ऑडियो

    if not text:
        return {"error": "Text is required"}

    try:
        ref_path = None
        active_prompt = None

        # 1. Preset Voice Mode (तीनों में से चुनी गई आवाज और उसका अपना प्रॉम्प्ट)
        if mode == 'preset':
            ref_path, active_prompt = get_preset_audio(preset_name)

        # 2. Custom User Upload Mode (यूजर की नई आवाज)
        elif mode == 'custom' and ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_path = tmp.name
            active_prompt = prompt_text if prompt_text else None

        if ref_path:
            wav = model.generate(
                text=text,
                prompt_wav_path=ref_path,
                prompt_text=active_prompt,
                cfg_value=2.0,
                inference_timesteps=10
            )
            if mode == 'custom' and os.path.exists(ref_path):
                os.unlink(ref_path)
        else:
            wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)

        # Output Base64 WAV
        buf = io.BytesIO()
        sr = getattr(model.tts_model, "sample_rate", 48000)
        sf.write(buf, wav, sr, format='WAV')
        out_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "audio_base64": out_b64
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
