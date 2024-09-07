

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
   git clone https://github.com/your-repo/ai-multifunction-app.git
   cd ai-multifunction-app
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

5. Run the Flask application:
   ```bash
   flask run
   ```

6. Open the application by navigating to `http://127.0.0.1:5000/` in your browser.

## Usage

### Text Chat
- Navigate to `/text-chat` and enter a message. The AI assistant will respond in plain text.

### Image Analysis
- Upload an image at `/image-analysis`, enter a prompt, and receive AI-generated insights.

### AI Chef
- Visit `/ai-chef` to upload an image of ingredients. The AI will suggest dishes and provide step-by-step instructions.

### Voice Interaction
- Visit `/voice-interaction` and start a voice conversation with the AI. The assistant listens and responds using text-to-speech.

### Language Translation
- Navigate to `/language-translation`, provide text, and select a target language for translation.

### Video Transcript Summarization
- Go to `/video-summarization`, enter a YouTube video link, and get a summarized transcript.

### Text-to-Speech
- Navigate to `/text-to-speech`, enter text, and convert it into speech.

## API Endpoints

### 1. `/text-chat` [POST]
- Input: JSON with a `text` field
- Output: JSON with the AI's response

### 2. `/image-analysis` [POST]
- Input: Image file and text prompt
- Output: AI-generated image analysis

### 3. `/ai-chef` [POST]
- Input: Image file of ingredients
- Output: JSON with a list of ingredients and suggested dishes

### 4. `/start-voice-interaction` [POST]
- Starts voice-based interaction with the AI.

### 5. `/stop-voice-interaction` [POST]
- Stops the ongoing voice interaction.

### 6. `/language-translation` [POST]
- Input: Text and target language
- Output: Translated text

### 7. `/video-summarization` [POST]
- Input: YouTube video URL
- Output: Summarized transcript

### 8. `/text-to-speech` [POST]
- Input: Text
- Output: Speech audio file

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or contributions, please contact:

- **Developer**: Sandeep Gudisa
- **Email**: sandeep.gudisa@domain.com

---

This `README.md` provides a comprehensive guide for anyone using or contributing to the project. You can customize the repository links and contact information.
