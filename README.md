

# Interactive AI

This AI-powered web application is designed to provide multiple functionalities such as text chat with an AI assistant, image analysis, AI chef suggestions, voice interaction, language translation, video transcript summarization, and text-to-speech conversion. The application leverages several AI models and services, including Google's Generative AI, ElevenLabs for text-to-speech, AssemblyAI for transcription, and YouTube transcript analysis.

## Features

### 1. Text Chat with AI
Engage in conversations with an AI assistant. The assistant processes user input and generates plain-text responses using Google's Generative AI model.

### 2. Image Analysis
Upload an image and provide a prompt to receive AI-generated responses based on the image. The AI analyzes the image and returns detailed insights or custom responses.

### 3. AI Chef Suggestions
Upload a photo of ingredients, and the AI will suggest dishes you can prepare with those ingredients. The AI also provides step-by-step cooking instructions.

### 4. Voice Interaction
Interact with the AI using voice commands. The AI listens, processes speech, responds in real-time, and plays the response back using text-to-speech.

### 5. Language Translation
Translate text into multiple languages using an integrated translation service. The system supports translations into a wide range of languages, with audio playback for the translated text.

### 6. Video Transcript Summarization
Provide a YouTube video link, and the AI will fetch the transcript and summarize the content using a text generation model.

### 7. Text-to-Speech Conversion
Convert any text into speech using Microsoft's text-to-speech technology and play the resulting audio.

## Tech Stack

- **Flask**: Web framework for Python
- **Google Generative AI**: For text and image processing
- **AssemblyAI**: For speech-to-text conversion
- **ElevenLabs**: For multilingual text-to-speech synthesis
- **YouTube Transcript API**: For extracting video transcripts from YouTube
- **Edge TTS**: For converting text to speech with Microsoft Azure
- **Pygame**: For playing audio responses

## Installation

### Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Google Generative AI API Key](https://developers.generative.ai/)
- [AssemblyAI API Key](https://www.assemblyai.com/)
- [ElevenLabs API Key](https://elevenlabs.io/)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GudisaSandeep/Interactive-AI/tree/master
   cd Interactive-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   Create a `.env` file in the project root directory and add the following:
   ```bash
   API_KEY=your_google_generative_ai_key
   ElevenLabs_api_key=your_elevenlabs_key
   api_key=your_assemblyai_key
   ```

4. Set up the file upload folder:
   ```bash
   mkdir uploads
   ```

5. Start Command
   ```bash
   python app.py
   ```

6. Open the application by navigating to `http://127.0.0.1:5000/` in your browser.



## License

This project is licensed under the MIT License.

## Contact

For any inquiries or contributions, please contact:

- **Developer**: Sandeep Gudisa
- **Email**: gudisasandeep141312@gmail.com


