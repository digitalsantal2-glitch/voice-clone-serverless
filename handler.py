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

print("Serverless AI Model लोड हो रहा है...")
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Model Load
llama_queue = launch_thread_safe_queue("/app/checkpoints/s2-pro", device=device, precision=precision, compile=False)
decoder_model = load_decoder_model(config_name="modded_dac_vq", checkpoint_path="/app/checkpoints/s2-pro/codec.pth", device=device)
engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=precision, compile=False)

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text')
    ref_audio_b64 = job_input.get('reference_audio')
    preset_name = job_input.get('preset_name', '')
    mode = job_input.get('mode', 'custom')

    if not text:
        return {"error": "Text is required"}

    try:
        audio_bytes = None
        if mode == 'custom' and ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)
        elif mode == 'preset' and preset_name:
            preset_path = os.path.join('/app/presets', preset_name)
            if os.path.exists(preset_path):
                with open(preset_path, 'rb') as f:
                    audio_bytes = f.read()

        if not audio_bytes:
            return {"error": "No reference audio provided"}

        req = ServeTTSRequest(text=text, references=[ServeReferenceAudio(audio=audio_bytes, text="")])
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
