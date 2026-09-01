import io
import os
import base64
import torch
import soundfile as sf
import runpod
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine import TTSInferenceEngine

print("Loading AI Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load S2-Pro Model
q = launch_thread_safe_queue("/app/checkpoints/s2-pro", device=device, precision=precision, compile=False)
dec = load_decoder_model(config_name="modded_dac_vq", checkpoint_path="/app/checkpoints/s2-pro/codec.pth", device=device)
engine = TTSInferenceEngine(llama_queue=q, decoder_model=dec, precision=precision, compile=False)
print("Model Ready!")

def handler(job):
    inp = job.get('input', {})
    text = inp.get('text', '')                    # नया डायलॉग
    prompt_text = inp.get('prompt_text', '')      # आवाज़ में जो बोला गया है
    ref_b64 = inp.get('reference_audio', '')      # ऑडियो फ़ाइल (Base64)

    if not text:
        return {"error": "Text is required"}
    if not ref_b64:
        return {"error": "Reference audio is required"}

    try:
        audio_bytes = base64.b64decode(ref_b64)

        # prompt_text के साथ 100% साफ आवाज़ बनेगी
        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=audio_bytes, text=prompt_text)]
        )

        for res in engine.inference(req):
            if res.code == "final":
                sr, audio_data = res.audio
                buf = io.BytesIO()
                sf.write(buf, audio_data, sr, format='WAV')
                return {
                    "status": "success",
                    "audio_base64": base64.b64encode(buf.getvalue()).decode('utf-8')
                }
            elif res.code == "error":
                return {"error": str(res.error)}
        return {"error": "Audio generation failed"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
