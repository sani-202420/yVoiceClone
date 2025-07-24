import gradio as gr
import torchaudio
import torch
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=torch.cuda.is_available())

def clone_voice(audio_file, text, speed, pitch):
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        speaker_embedding = tts.get_speaker_embedding(audio_file)
        wav = tts.tts(text, speaker_wav=audio_file, speaker_embedding=speaker_embedding, speed=speed)
        out_path = "output.wav"
        tts.save_wav(wav, out_path)
        return out_path
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("# 🎤 yVoiceClone - AI Voice Cloner")
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="🗣️ Upload or Record Voice", type="filepath", source="upload", interactive=True)
            input_text = gr.Textbox(label="✍️ Enter text to speak in cloned voice")
            speed = gr.Slider(0.5, 2.0, value=1.0, label="⏩ Speed")
            pitch = gr.Slider(-5, 5, value=0, label="🎚️ Pitch (optional)")
            clone_btn = gr.Button("🔊 Clone Voice")
        with gr.Column():
            output_audio = gr.Audio(label="📢 Cloned Voice Output")

    clone_btn.click(fn=clone_voice, inputs=[input_audio, input_text, speed, pitch], outputs=output_audio)

app.launch()
