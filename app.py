from flask import Flask, request, render_template, jsonify, url_for, send_file, flash
import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import io
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import threading
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
import base64
import re
import time

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if API_KEY is None:
    @app.route('/')
    def api_key_error():
        return "API Key is not set. Please set the API key in the .env file.", 500
else:
    genai.configure(api_key=API_KEY)

def transcribe_audio(audio_file):
    aai.settings.api_key = os.getenv("api_key")
    transcriber = aai.Transcriber()
    return transcriber.transcribe(audio_file)

def text_chat(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = text + "\n\nProvide your response in plain text format without any markdown, formatting, or special characters."
    response = model.generate_content(
        glm.Content(
            parts=[glm.Part(text=prompt)],
        ),
        stream=True
    )
    response.resolve()
    return {"You": text, "Assistant": response.text}

def cleanup_old_audio_files(directory, max_age_seconds=3600):
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.mp3'):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)

def image_analysis(image, prompt):
    pil_image = Image.open(io.BytesIO(image))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    bytes_data = img_byte_arr.getvalue()

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=prompt + "\n\nProvide your response in plain text format without any markdown, formatting, or special characters."),
                glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
            ],
        ),
        stream=True
    )
    response.resolve()
    return response.text

def ai_chef_analysis(image):
    pil_image = Image.open(io.BytesIO(image))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    bytes_data = img_byte_arr.getvalue()

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    chef_prompt = (
        "Analyze this image and identify the ingredients present. "
        "Then, suggest 3 dishes that can be made using these ingredients.\n\n"
        "Format your response as follows:\n\n"
        "Ingredients: [List of identified ingredients]\n\n"
        "Suggested Dishes:\n"
        "1. [Dish name]: [Brief description]\n\n"
        "2. [Dish name]: [Brief description]\n\n"
        "3. [Dish name]: [Brief description]\n\n"
        "Give the process of making step by step\n\n"
        "Provide your response in plain text format without any markdown, formatting, or special characters."
    )
    
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=chef_prompt),
                glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
            ],
        ),
        stream=True
    )
    response.resolve()
    return response.text

def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

async def text_to_speech(text, voice):
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        if audio_data.getvalue():
            return audio_data.getvalue()
        else:
            return None
    except edge_tts.exceptions.NoAudioReceived:
        print(f"No audio received for text: '{text}' with voice: '{voice}'")
        return None

class VoiceInteraction:
    def __init__(self):
        self.is_running = False
        self.recognizer = sr.Recognizer()
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.conversation = []

    async def text_to_speech_and_play(self, text):
        try:
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            audio_path = "output.mp3"
            await communicate.save(audio_path)

            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def listen_and_respond(self):
        with sr.Microphone() as source:
            while self.is_running:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized text: {text}")
                    self.conversation.append({"You": text})

                    response = self.model.generate_content(
                        glm.Content(parts=[glm.Part(text=text)]),
                        stream=True
                    )
                    response.resolve()
                    print(f"AI Response: {response.text}")
                    self.conversation.append({"Assistant": response.text})

                    asyncio.run(self.text_to_speech_and_play(response.text))
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Request error: {str(e)}")
                except Exception as e:
                    print(f"General error: {e}")

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.listen_and_respond)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()

voice_interaction = VoiceInteraction()

def process_input(text_input, audio_file, recorded_audio):
    if text_input:
        return translate_and_generate(text_input)
    elif audio_file:
        transcript = transcribe_audio(audio_file)
    elif recorded_audio:
        transcript = transcribe_audio(recorded_audio)
    else:
        raise ValueError("Please provide either text input or an audio file.")
    
    return translate_and_generate(transcript.text)

def translate_and_generate(text):
    list_translations = translate_text(text)
    generated_audio_paths = []

    for translation in list_translations:
        translated_audio_file_name = multi_language_text_to_speech(translation)
        path = Path(translated_audio_file_name)
        generated_audio_paths.append(str(path))

    return {
        "audio_paths": generated_audio_paths,
        "translations": list_translations
    }

def translate_text(text):
    languages = ["ru", "tr", "sv", "de", "es", "ja", "hi", "te"]
    list_translations = []

    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)

    return list_translations

def multi_language_text_to_speech(text):
    client = ElevenLabs(api_key=os.getenv("ElevenLabs_api_key"))

    audio = client.generate(
        text=text,
        voice="nPczCjzI2devNBz1zQrb",
        model="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        f.write(audio["audio_data"])

    return save_file_path

@app.route('/text', methods=['POST'])
def text_route():
    text_input = request.form.get('text_input')
    if text_input:
        result = process_input(text_input, None, None)
        return jsonify(result)
    return jsonify({"error": "No text input provided"}), 400

@app.route('/audio', methods=['POST'])
def audio_route():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400

    if audio_file:
        result = process_input(None, audio_file, None)
        return jsonify(result)
    return jsonify({"error": "Audio file upload failed"}), 500

@app.route('/recorded', methods=['POST'])
def recorded_route():
    if 'recorded_audio' not in request.files:
        return jsonify({"error": "No recorded audio provided"}), 400

    recorded_audio = request.files['recorded_audio']
    if recorded_audio.filename == '':
        return jsonify({"error": "No selected recorded audio"}), 400

    if recorded_audio:
        result = process_input(None, None, recorded_audio)
        return jsonify(result)
    return jsonify({"error": "Recorded audio upload failed"}), 500

@app.route('/image', methods=['POST'])
def image_route():
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image_file']
    prompt = request.form.get('prompt', '')
    
    if image_file and image_file.filename != '':
        image = image_file.read()
        result = image_analysis(image, prompt)
        return jsonify({"result": result})
    return jsonify({"error": "Image file upload failed"}), 500

@app.route('/ai-chef', methods=['POST'])
def ai_chef_route():
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image_file']
    if image_file and image_file.filename != '':
        image = image_file.read()
        result = ai_chef_analysis(image)
        return jsonify({"result": result})
    return jsonify({"error": "Image file upload failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
