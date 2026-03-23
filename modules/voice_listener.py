import whisper
import sounddevice as sd
import numpy as np
import threading
import queue

class VoiceListener:
    def __init__(self, sample_rate=16000, model_name="base"):
        self.sample_rate = sample_rate
        # "base" model is a good tradeoff between speed and accuracy for real-time
        self.model = whisper.load_model(model_name) 
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_listening = False

        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            dtype='float32', 
            callback=self.audio_callback
        )
        # buffer for 2.5 seconds of audio to capture short jutsu names like "Fireball Jutsu"
        self.buffer = np.zeros((0, 1), dtype='float32')
        self.buffer_size = int(self.sample_rate * 2.5)

    def audio_callback(self, indata, frames, time, status):
        if self.is_listening:
            self.buffer = np.append(self.buffer, indata, axis=0)
            if len(self.buffer) >= self.buffer_size:
                # Push the 2.5s chunk to queue and reset buffer
                audio_chunk = self.buffer.copy().flatten()
                self.audio_queue.put(audio_chunk)
                
                # Keep the last 0.5s to overlap so we don't cut words in half
                overlap_size = int(self.sample_rate * 0.5)
                self.buffer = self.buffer[-overlap_size:]

    def process_audio(self):
        while self.is_listening:
            try:
                # Wait up to 1s for audio
                audio_data = self.audio_queue.get(timeout=1)
                
                # Transcribe the numpy array
                result = self.model.transcribe(audio_data, fp16=False, language="en")
                text = result.get('text', '').strip().lower()
                
                if text:
                    self.result_queue.put(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceListener] Error: {e}")

    def start(self):
        self.is_listening = True
        self.stream.start()
        self.thread = threading.Thread(target=self.process_audio, daemon=True)
        self.thread.start()
        print("[VoiceListener] Started listening...")

    def stop(self):
        self.is_listening = False
        self.stream.stop()
        self.stream.close()
        print("[VoiceListener] Stopped.")

    def get_latest_phrase(self):
        """Returns the most recently transcribed text (non-blocking)."""
        phrases = []
        while not self.result_queue.empty():
            phrases.append(self.result_queue.get())
        
        if phrases:
            return " ".join(phrases)
        return None
