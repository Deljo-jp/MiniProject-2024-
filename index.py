import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai
import os
import threading
import customtkinter as ctk
from tkinter import messagebox
import warnings
warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")

from faster_whisper import WhisperModel

whispersize = 'base'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whispersize,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores,
)

GOOGLE_API_KEY = 'Enter Your API key here'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    'gemini-1.0-pro-latest',
    generation_config=generation_config,
    safety_settings=safety_settings
)

convo = model.start_chat()

# System message
system_message = '''Give short replies, don't reply "short reply" back'''
system_message = system_message.replace('\n', '')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()  # Initialize source immediately

listening = False  # Global variable to control listening state
listener_thread = None

def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    
    audio = AudioSegment.from_mp3(filename)
    play(audio)
    
    os.remove(filename)

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def callback(recognizer, audio):
    global listening  # Reference the global variable
    if not listening:
        return
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)

        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
            update_status('Empty prompt. Please speak again.')
        else:
            print('User:', prompt_text.encode('utf-8', 'ignore').decode('utf-8'))
            update_conversation("User: " + prompt_text + "\n")

            convo.send_message(prompt_text)
            output = convo.last.text

            if output.lower().strip() != "hello":
                print('Gemini:', output.encode('utf-8', 'ignore').decode('utf-8'))
                update_conversation("Gemini: " + output + "\n")
                speak(output)
            else:
                print("Gemini: You're welcome. Is there anything else I can help you with today?")
                update_conversation("Gemini: You're welcome. Is there anything else I can help you with today?\n")

    except Exception as e:
        print('Prompt error:', e)
        update_status(f'Prompt error: {e}')

def start_listening_thread():
    global listener_thread
    stop_listening_thread()  # Stop any existing listener before starting a new one
    listener_thread = threading.Thread(target=start_listening)
    listener_thread.start()
    print("Voice interaction started...")
    update_status("Voice interaction started...")

def start_listening():
    global listening
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('Listening...')
    update_status('Listening...')

    listening = True
    r.listen_in_background(source, callback)

def stop_listening_thread():
    global listener_thread
    if listener_thread and listener_thread.is_alive():
        stop_listening()
        listener_thread.join()
        print("Stopped listening.")
        update_status("Stopped listening.")

def stop_listening():
    global listening
    listening = False

def update_conversation(message):
    output_text.configure(state=ctk.NORMAL)
    output_text.insert(ctk.END, message)
    output_text.configure(state=ctk.DISABLED)
    output_text.see(ctk.END)

def update_status(message):
    output_text.configure(state=ctk.NORMAL)
    output_text.insert(ctk.END, message + "\n")
    output_text.configure(state=ctk.DISABLED)
    output_text.see(ctk.END)

def type_and_get_reply():
    user_input = input_text.get("1.0", ctk.END).strip()
    if len(user_input) > 0:
        print('User:', user_input.encode('utf-8', 'ignore').decode('utf-8'))
        update_conversation("User: " + user_input + "\n")
        convo.send_message(user_input)
        output = convo.last.text
        print('Gemini:', output.encode('utf-8', 'ignore').decode('utf-8'))
        update_conversation("Gemini: " + output + "\n")
    else:
        messagebox.showwarning("Input Error", "Please enter a prompt.")
    input_text.delete("1.0", ctk.END)

# GUI setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Gemini Voice Assistant")
root.geometry("800x600")  # Adjusted size for normal PC screen

frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=20)

input_label = ctk.CTkLabel(frame, text="Type your prompt to Gemini:")
input_label.grid(row=0, column=0, padx=10, pady=5)

input_text = ctk.CTkTextbox(frame, width=600, height=200)
input_text.grid(row=1, column=0, padx=10, pady=5)

submit_button = ctk.CTkButton(frame, text="Submit", command=type_and_get_reply)
submit_button.grid(row=2, column=0, padx=10, pady=5)

voice_button = ctk.CTkButton(frame, text="Start Voice Interaction", command=start_listening_thread)
voice_button.grid(row=3, column=0, padx=10, pady=5)

stop_voice_button = ctk.CTkButton(frame, text="Stop Voice Interaction", command=stop_listening_thread)
stop_voice_button.grid(row=4, column=0, padx=10, pady=5)

output_label = ctk.CTkLabel(frame, text="Conversation:")
output_label.grid(row=5, column=0, padx=10, pady=5)

output_text = ctk.CTkTextbox(frame, width=600, height=200, wrap=ctk.WORD, state=ctk.DISABLED)
output_text.grid(row=6, column=0, padx=10, pady=5)

root.mainloop()