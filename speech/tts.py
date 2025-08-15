from gtts import gTTS
from playsound import playsound

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")  # plays inside Python process
