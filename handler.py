import io
import os
import base64
import soundfile as sf
import torch
import runpod
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.inference_engine import TTSInferenceEngine

print("1. Loading Fish Speech 1.5 Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load Fish Speech 1.5 Stable
llama_queue = launch_thread_safe_queue(
    checkpoint_path="/app/checkpoints/fish-speech-1.5",
    device=device,
    precision=precision,
    compile=False
)

decoder_model = load_decoder_model(
    config_name="firefly_gan_vq",
    checkpoint_path="/app/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    device=device
)

engine = TTSInferenceEngine(
    llama_queue=llama_queue,
    decoder_model=decoder_model,
    precision=precision,
    compile=False
)
print("2. Fish Speech 1.5 Ready!")

COMMON_PROMPT = "मेरे हौसले की उड़ान अब कम नहीं होगी, मेरे जिद की जिद खत्म नहीं होगी। अब पूरा हिंदुस्तान जीतूंगा सिकंदर बनकर। यह ऐलान-ए-जंग खुली आम होगा, पूरे इंडिया में, सरेआम होगा! एक दमदार प्रस्तुति के लिए तैयार हो जाइए।"

PRESETS = {
    "voice1": {
        "file": "/app/presets/long_kolhapuri.wav",
        "prompt": COMMON_PROMPT
    },
    "voice2": {
        "file": "/app/presets/competition_dialogue.mp3",
        "prompt": COMMON_PROMPT
    },
    "voice3": {
        "file": "/app/presets/competition_voice.mp3",
        "prompt": COMMON_PROMPT
    }
}

def handler(job):
    inp = job.get('input', {})
    text = inp.get('text', '')                    # नया डायलॉग
    mode = inp.get('mode', 'preset')              # 'preset' या 'custom'
    preset_name = inp.get('preset_name', 'voice1')# 'voice1', 'voice2', 'voice3'
    prompt_text = inp.get('prompt_text', '')
    ref_audio_b64 = inp.get('reference_audio', '')

    if not text:
        return {"error": "Text is required"}

    try:
        audio_bytes = None
        clean_prompt = prompt_text

        # 1. Preset Voice
        if mode == 'preset':
            preset_info = PRESETS.get(preset_name, PRESETS["voice1"])
            with open(preset_info["file"], "rb") as f:
                audio_bytes = f.read()
            clean_prompt = preset_info["prompt"]

        # 2. Custom User Upload
        elif mode == 'custom' and ref_audio_b64:
            audio_bytes = base64.b64decode(ref_audio_b64)

        if not audio_bytes:
            return {"error": "No audio provided"}

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
                    "audio_base64": base64.b64encode(buf.getvalue()).decode('utf-8')
                }
            elif result.code == "error":
                return {"error": str(result.error)}

        return {"error": "Inference failed"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
