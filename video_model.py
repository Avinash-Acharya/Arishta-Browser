import os
import json
import time
import torch
import whisper
import subprocess
from summarizer import summarize
from nvidiaNim import model_2_1, model_2_2
from news_fakery import fake_video_detector
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# S2T_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Stokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_ID)
# Smodel = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_ID)
model = whisper.load_model("base").to(device)
# Smodel.to(device)

def download_audio(url, output_file, max_duration=100):

    print("- Downloading audio...")

    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print-json", url],
        stdout=subprocess.PIPE, text=True
    )
    video_info = json.loads(result.stdout)
    duration = video_info["duration"]  
    download_duration = min(max_duration, duration)
    subprocess.run([
        "yt-dlp", url,
        "-x", "--audio-format", "mp3",
        f"--postprocessor-args=-ss 00:00:00 -t {download_duration}",
        "-o", output_file
    ])

def transcribe_audio(audio_file):

    print("- Transcribing audio...")
    result = model.transcribe(audio_file)

    return result["text"]  

def summarize_text(text):

    print("- Summarizing text...")

    if s_model == 1:
        summarized = summarize(text)
    elif s_model == 2:
        summarized = model_2_1(text)
    elif s_model == 3:
        summarized = model_2_2(text)
    return summarized

def fake_video_news(url, modelNo):

    global s_model
    s_model = modelNo
    print("- Processing video...")
    start_time = time.time()
    output_file = "./audio.mp3"
    delete_after_process = True 
    download_audio(url, output_file)
    transcript = transcribe_audio(output_file)
    summary = summarize_text(transcript)
    result = fake_video_detector(summary)

    if delete_after_process and os.path.exists(output_file):
        os.remove(output_file)
        print(f"File {output_file} has been deleted.")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to analyze: {time_taken:.2f} seconds")
    
    return result

