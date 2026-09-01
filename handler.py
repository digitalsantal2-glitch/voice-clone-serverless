import io
import os
import base64
import tempfile
import soundfile as sf
import runpod
from voxcpm import VoxCPM

print("Loading OpenBMB VoxCPM Model...")
# Official OpenBMB VoxCPM Model
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("OpenBMB VoxCPM Ready!")

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text', '')                    # नया डायलॉग
    prompt_text = job_input.get('prompt_text', '')      # Voice फ़ाइल में बोला गया exact text
    ref_audio_b64 = job_input.get('reference_audio', '')# आपकी ऑडियो फ़ाइल

    if not text:
        return {"error": "Text is required"}

    try:
        ref_path = None
        if ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_path = tmp.name

        # OpenBMB VoxCPM Voice Cloning
        if ref_path:
            wav = model.generate(
                text=text,
                prompt_wav_path=ref_path,
                prompt_text=prompt_text if prompt_text else None,
                cfg_value=2.0,
                inference_timesteps=10
            )
            os.unlink(ref_path)
        else:
            wav = model.generate(
                text=text,
                cfg_value=2.0,
                inference_timesteps=10
            )

        # Output to Base64 WAV
        buf = io.BytesIO()
        sample_rate = getattr(model.tts_model, "sample_rate", 48000)
        sf.write(buf, wav, sample_rate, format='WAV')
        out_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "audio_base64": out_b64
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
