import queue
import torch
import librosa
import numpy as np
import pyaudio
import time
import spacy
from PyQt5.QtCore import QThread, pyqtSignal

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from googletrans import Translator

from pysentimiento import create_analyzer

from dask.distributed import Client

# Parametros de Pyaudio
CHUNK = 1024              # Numero de frames por buffer
FORMAT = pyaudio.paInt16  # _Formato de captura de aurio_
CHANNELS = 1              # Numero de canales de audio (mono)
RATE = 16000              # Tasa de sonido (16 kHz es necesario para Whisper)
RECORD_SECONDS = 15       # Duracion de cada grabacion de chunk de audio

lenguaje_analyzer = spacy.load("es_core_news_md") # Para comparar los textos

modelo_huggingFace = "openai/whisper-small" # "openai/whisper-large-v3-turbo" # 
parallelType = "cuda" if torch.cuda.is_available() else "cpu" # CUDA para operacion con gpu
model = WhisperForConditionalGeneration.from_pretrained(modelo_huggingFace).to(parallelType) # Modelo para transformar audio en texto
processor = WhisperProcessor.from_pretrained(modelo_huggingFace) # Procesador para _tokenizar_ el audio
forced_decoder_ids = processor.get_decoder_prompt_ids(language="es", task="transcribe") # Configuraciones del modelo, "en" Ingles, "es" Espanol

sentimientos_analyzer = create_analyzer(task="sentiment", lang="en") # Analizador de sentimientos

traductor = Translator() # Traductor de idiomas

def work_obtener_audio(audio_queue, pyaudioref, iniciado_flag):
        while iniciado_flag["value"]:
            '''Graba el audio del microfono por un lapso de tiempo de RECORD_SECONDS, retorna un array de numpy'''
            stream = pyaudioref.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []

            print("Recording...")
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))

            print("Recording finished.")
            stream.stop_stream()
            stream.close()
            
            # "Combinar frames" para normalizar el array de audio
            audio_data = np.hstack(frames).astype(np.float32) # en promedio son algo de 159744 slots
            audio_data /= np.max(np.abs(audio_data))  # Normalizar el audio para que este entre -1 y 1

            audio_queue.put(audio_data)


def procesar_pipeline(audio_queue, iniciado_flag):
    while iniciado_flag["value"]:
        try:
            audio_data = audio_queue.get(timeout=1)

            # Process the audio data
            # Procesa el audio para que concuerde con los requerimientos del modelo
            input_features = processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_features.to(parallelType)
            
            # Generar transcripcion con "forced language decoding" (genera tokens)
            generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
            # Convierte los tokens del audio a texto
            transcription_actual = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Traducir texto
            texto_traducido = traductor.translate(transcription_actual, src='es', dest='en').text
            
            # Emociones
            emocion = sentimientos_analyzer.predict(transcription_actual)
            emocion_actual = emocion.output + " " + str('%.3f'%max(emocion.probas.values()))

            # Output the results
            print(f"Transcripción: {transcription_actual}")
            print(f"Traducción: {texto_traducido}")
            print(f"Emoción: {emocion_actual}")

        except queue.Empty:
            print("Lista vacia, pasando al siguiente")
            time.sleep(0.1)
            #No pasa nada de esar vacia
            continue

class AnalisisVoz_Paralelo_ProCons_Dask():

    def __init__(self, pyaudio_ref):
        self.p = pyaudio_ref

        self.iniciado = False
        self.audio_queue = queue.Queue()    

    def proceso_main(self):
        """Funcion para iniciar la transcripción usando el microfono como constante entrada de audio"""
        self.iniciado = True

        iniciado_flag = {"value": True}

        # Start dask client
        client = Client()

        try:
            future_audio = client.submit(
                work_obtener_audio,
                self.audio_queue,
                self.p,
                iniciado_flag
            )
            future_pipeline = client.submit(
                procesar_pipeline,
                self.audio_queue,
                iniciado_flag
            )

            while self.iniciado:
                time.sleep(0.1)
                continue

        except KeyboardInterrupt:
            print("Deteniendo transcripción desde el micrófono.")
            self.iniciado = False
            iniciado_flag["value"] = False
        
        finally:
            future_audio.cancel()
            future_pipeline.cancel()
            client.close()


# Funcion Main
if __name__ == "__main__":

    p_audio = pyaudio.PyAudio()

    TranscripcionManager = AnalisisVoz_Paralelo_ProCons_Dask(p_audio)

    audio_file_path = 'audiosTest/'

    #TranscripcionManager.analizar_desde_microfono()
    '''
    audio_files = [os.path.join(audio_file_path, f) for f in os.listdir(audio_file_path) if f.endswith(('.wav', '.mp3'))]
    for file_path in audio_files:
        TranscripcionManager.test_clase(file_path)
    '''
    TranscripcionManager.proceso_main()