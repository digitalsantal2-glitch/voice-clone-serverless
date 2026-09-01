import io
import os
import base64
import tempfile
import numpy as np
import soundfile as sf
import torch
import runpod
from voxcpm import VoxCPM

print("Loading OpenBMB VoxCPM2 Model...")
# Clean Voice Model with Denoiser
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=True)
print("OpenBMB VoxCPM2 Ready!")

PRESETS = {
    "voice1": "/app/presets/long_kolhapuri.wav",
    "voice2": "/app/presets/competition_dialogue.mp3",
    "voice3": "/app/presets/competition_voice.mp3"
}

def handler(job):
    inp = job.get('input', {})
    text = inp.get('text', '')                    # नया डायलॉग
    mode = inp.get('mode', 'preset')              # 'preset' या 'custom'
    preset_name = inp.get('preset_name', 'voice1')# चुनी गई आवाज
    prompt_text = inp.get('prompt_text', '')      # कस्टम प्रॉम्प्ट
    ref_audio_b64 = inp.get('reference_audio', '')# कस्टम ऑडियो

    if not text:
        return {"error": "Text is required"}

    try:
        ref_path = None

        if mode == 'preset':
            ref_path = PRESETS.get(preset_name, PRESETS["voice1"])
        elif mode == 'custom' and ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_path = tmp.name

        # 👉 Pure Timbre Voice Cloning (बिना किसी बैकग्राउंड शोर के साफ़ आवाज़)
        if ref_path and os.path.exists(ref_path):
            if prompt_text:
                wav = model.generate(
                    text=text,
                    prompt_wav_path=ref_path,
                    prompt_text=prompt_text,
                    cfg_value=1.8,
                    inference_timesteps=15
                )
            else:
                wav = model.generate(
                    text=text,
                    reference_wav_path=ref_path,
                    cfg_value=1.8,
                    inference_timesteps=15
                )

            if mode == 'custom':
                os.unlink(ref_path)
        else:
            wav = model.generate(text=text, cfg_value=1.8, inference_timesteps=15)

        # 🎚️ Audio Peak Normalization (आवाज़ के फटने और गर्जना को रोकने के लिए)
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy().squeeze()
        elif isinstance(wav, np.ndarray):
            wav = wav.squeeze()

        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val

        # 48kHz Studio Quality WAV
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
