
from flask import Flask, request, render_template, jsonify, url_for, send_file,flash
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
import re,time

API_KEY = "sk_9852342ec7afeaed89f471384e7e8cf019cd2899167704e2"

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if API_KEY is None:
    @app.route('/')
    def api_key_error():
        return "API Key is not set. Please set the API key in the .env file."
else:
    genai.configure(api_key=API_KEY)
def transcribe_audio(audio_file):
    aai.settings.api_key = "551438c8eb2d4eb2927307d8c763408e"
    transcriber = aai.Transcriber()
    return transcriber.transcribe(audio_file)


def text_chat(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = text + "\n\nProvide your response in plain text format without any markdown, formatting, or special characters."
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=prompt),
            ],
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
                glm.Part(text=prompt +  "\n\nProvide your response in plain text format without any markdown, formatting, or special characters."),
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
        "Analyze this image and identify the ingredients present.   "
        "Then, suggest 3 dishes that can be made using these ingredients. + \n\n "
        "Format your response as follows: + \n\n"
        "Ingredients: [List of identified ingredients]"
        "Suggested Dishes:"
        "1. [Dish name]: [Brief description] + \n\n"
        "2. [Dish name]: [Brief description] + \n\n"
        "3. [Dish name]: [Brief description] + \n\n"
        "Give the process of making step by step" + "\n\nProvide your response in plain text format without any markdown, formatting, or special characters."
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

# Asynchronous text-to-speech function
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
    client = ElevenLabs(
        api_key="sk_24a253c49ed39f920f9849c7dae74ab26d6b8c826089b0c6",
    )

    audio = generate(
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
        f.write(audio)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-chat', methods=['GET', 'POST'])
def handle_text_chat():
    if request.method == 'POST':
        text = request.form['text']
        result = text_chat(text)
        return jsonify(result)
    return render_template('text_chat.html')

@app.route('/image-analysis', methods=['GET', 'POST'])
def handle_image_analysis():
    if request.method == 'POST':
        image = request.files['image'].read()
        prompt = request.form['prompt']
        result = image_analysis(image, prompt)
        return jsonify(result)
    return render_template('image_analysis.html')
@app.route('/ai-chef', methods=['GET', 'POST'])
def ai_chef():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image = request.files['image'].read()
        result = ai_chef_analysis(image)
        
        # Parse the result to separate ingredients and suggestions
        parts = result.split("Suggested Dishes:")
        ingredients = parts[0].replace("Ingredients:", "").strip()
        suggestions = parts[1].strip() if len(parts) > 1 else "No suggestions available."
        
        return jsonify({
            "ingredients": ingredients,
            "suggestions": suggestions
        })
    
    return render_template('AI_Chef.html')

@app.route('/start-voice-interaction', methods=['POST'])
def start_voice_interaction():
    voice_interaction.start()
    return jsonify({"status": "Voice interaction started. Speak now!"})

@app.route('/stop-voice-interaction', methods=['POST'])
def stop_voice_interaction():
    voice_interaction.stop()
    return jsonify({"status": "Voice interaction stopped.", "conversation": voice_interaction.conversation})

@app.route('/voice-interaction', methods=['GET'])
def voice_interaction_page():
    return render_template('voice_interaction_page.html')

@app.route('/language-translation', methods=['GET', 'POST'])
def language_translation():
    if request.method == 'POST':
        text = request.form['text']
        target_language = request.form['target_language']
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return jsonify({"translated_text": translated.text})
    return render_template('language_translation.html')

def summarize_video_transcript(transcript):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"zSummarize the following video transcript:\n\n{transcript}"
    response = model.generate_content(
        glm.Content(parts=[glm.Part(text=prompt)]),
        stream=True
    )
    response.resolve()
    return response.text

@app.route('/video-summarization', methods=['GET', 'POST'])
def video_summarization():
    if request.method == 'POST':
        video_url = request.form['video_url']
        video_id = video_url.split('v=')[-1]
        transcript = get_video_transcript(video_id)
        summary = summarize_video_transcript(transcript)
        return jsonify({"summary": summary})
    return render_template('video_summarozation.html')

@app.route('/text-to-speech', methods=['GET', 'POST'])
def text_to_speech_route():
    if request.method == 'POST':
        text = request.form['text']
        audio_file = text_to_speech(text)
        return jsonify({"audio_file": audio_file})
    return render_template('text_to_speech.html')


@app.route('/kids-ai', methods=['GET', 'POST'])
def kids_ai():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image'].read()
            prompt = "You are a friendly AI assistant for kids. Analyze this image and describe it in a fun, educational way for children. Include interesting facts and ask an engaging question about the image."
            result = image_analysis(image, prompt)
        else:
            homework = request.form['homework']
            prompt = f"You are a friendly AI tutor for kids. Help with this homework in a fun and encouraging way: {homework}. Use simple language, provide step-by-step explanations, and include a fun fact related to the topic."
            result = text_chat(prompt)
        
        # Extract only the text response from the result
        if isinstance(result, dict) and 'Assistant' in result:
            response_text = result['Assistant']
        else:
            response_text = str(result)
        
        return jsonify({"response": response_text})
    return render_template('kids_ai.html')
@app.route('/content-generator', methods=['GET', 'POST'])
def content_generator():
    if request.method == 'POST':
        topic = request.form['topic']
        content_type = request.form['content_type']
        prompt = f"Generate {content_type} ideas for a YouTube video on: {topic}. Include title, description, and key points. Provide the response in plain text format without any markdown or special formatting."
        generated_content = text_chat(prompt)
        
        # Remove any remaining markdown or formatting
        plain_text = re.sub(r'[#*_\-\[\]()]', '', generated_content['Assistant'])
        plain_text = re.sub(r'\n+', '\n', plain_text).strip()
        
        return jsonify({"Assistant": plain_text})
    return render_template('content_generator.html')

@app.route('/interactive-ai', methods=['GET', 'POST'])
def interactive_ai():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = text_chat(user_input)
        return jsonify(response)
    return render_template('interactive_ai.html')


@app.route('/code-generator', methods=['GET', 'POST'])
def code_generator():
    if request.method == 'POST':
        code_request = request.form['code_request']
        language = request.form['language']
        prompt = f"Generate {language} code for: {code_request}"
        generated_code = text_chat(prompt)
        return jsonify(generated_code)
    return render_template('code_generator.html')
@app.route('/TTS', methods=['GET', 'POST'])
def TTS():
    VOICE_OPTIONS = {
    "English": {
        "en-AU-NatashaNeural": "Natasha (Australia)",
        "en-CA-LiamNeural": "Liam (Canada)",
        # Add more voices as in the previous Streamlit code
    },
    "spanish": {
        "es-ES-ElviraNeural": "Elvira (Spain)",
        "es-MX-JorgeNeural": "Jorge (Mexico)"

    },
    "french": {
        "fr-CA-JeanNeural": "Jean (Canada)",
        "fr-FR-DeniseNeural": "Denise (France)"
    },
    "german": {
        "de-AT-IngridNeural": "Ingrid (Austria)",
        "de-DE-ConradNeural": "Conrad (Germany)"

    },
    "italian": {
        
        "it-IT-ElsaNeural": "Elsa (Italy)"

    },
    "portuguese": {
        "pt-BR-FranciscaNeural": "Francisca (Brazil)",
        "pt-PT-DuarteNeural": "Duarte (Portugal)"
    },
    "telugu" : {
        "te-IN-PriyaNeural": "Priya (India)",
        "te-IN-SakethNeural": "Saketh (India)",

    },
    "Hindhi": {
        "hi-IN-MadhurNeural": "Madhur (India)",
        "hi-IN-SwaraNeural": "Swara (India)"

    },
    "Tamil": {
        "ta-IN-PallaviNeural": "Pallavi (India)",
        "ta-IN-ValluvarNeural": "Valluvar (India)"
    },
    "Malayalam": {
        "ml-IN-MidhunNeural": "Midhun (India)",
        "ml-IN-SobhanaNeural": "Sobhana (India)"

    },
    "Kannada": {
        "kn-IN-GaganNeural": "Gagan (India)",
        "kn-IN-SakethNeural": "Saketh (India)"
    },
    "Marathi": {
        "mr-IN-SakshiNeural": "Sakshi (India)",
        "mr-IN-VaishaliNeural": "Vaishali (India)"

    },
    "Gujarati": {
        "gu-IN-DhwaniNeural": "Dhwani (India)",
        "gu-IN-NiranjanNeural": "Niranjan (India)"
    },
    "Bengali": {
        
        "bn-IN-TanishqNeural": "Tanishq (India)",
        "bn-IN-BashkarNeural": "Bashkar (India)"
    },
    "Korean": {
        "ko-KR-InJoonNeural": "InJoon (Korea)",
        "ko-KR-SeoyeonNeural": "Seoyeon (Korea)"
    },
    "Japanese": {
        "ja-JP-NanamiNeural": "Nanami (Japan)",
        "ja-JP-KeitaNeural": "Keita (Japan)"
    },
    "Chinese": {
        "zh-CN-XiaoxiaoNeural": "Xiaoxiao (China)",
        "zh-CN-XiaoyouNeural": "Xiaoyou (China)"
    },}
    languages = list(VOICE_OPTIONS.keys())
    selected_language = request.form.get("language", "English")
    selected_voice = request.form.get("voice", next(iter(VOICE_OPTIONS[selected_language].keys())))
    audio_data = None
    
    if request.method == 'POST':
        text_input = request.form.get("text_input")
        if text_input:
            audio_data = asyncio.run(text_to_speech(text_input, selected_voice))
            if audio_data is None:
                flash("No audio could be generated. Please try again with different text or voice settings.", "error")

    return render_template("TTS.html", 
                           languages=languages, 
                           selected_language=selected_language, 
                           selected_voice=selected_voice, 
                           audio_data=audio_data, 
                           voice_options=VOICE_OPTIONS)
@app.route('/download')
def download_audio():
    audio_data = request.args.get("audio")
    audio_file = io.BytesIO(base64.b64decode(audio_data))
    audio_file.seek(0)
    return send_file(audio_file, mimetype="audio/mp3", as_attachment=True, download_name="generated_speech.mp3")

@app.template_filter('b64encode')
def b64encode_filter(s):
    return base64.b64encode(s).decode('utf-8')

@app.route('/about-us')
def about_us():
    return render_template('About us.html')


@app.route('/multi_language_translator', methods=['GET', 'POST'])
def multi_language_translator():
    if request.method == 'POST':
        text_input = request.form.get('text_input')
        audio_file = request.files.get('audio_file')
        recorded_audio = request.files.get('recorded_audio')

        try:
            result = process_input(text_input, audio_file, recorded_audio)
            # Convert audio paths to base64 encoded data
            for i, path in enumerate(result['audio_paths']):
                with open(path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                    result['audio_paths'][i] = base64.b64encode(audio_data).decode('utf-8')
            
            # Add this at the end of the POST handling
            #cleanup_old_audio_files('static/audio', max_age_seconds=3600)  # Clean up files older than 1 hour
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template('multi_language_translator.html')
@app.route('/get-conversation', methods=['GET'])
def get_conversation():
    return jsonify({"conversation": voice_interaction.conversation})





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
