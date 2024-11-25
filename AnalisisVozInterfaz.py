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

# Parametros de Pyaudio
CHUNK = 1024              # Numero de frames por buffer
FORMAT = pyaudio.paInt16  # _Formato de captura de aurio_
CHANNELS = 1              # Numero de canales de audio (mono)
RATE = 16000              # Tasa de sonido (16 kHz es necesario para Whisper)
RECORD_SECONDS = 4       # Duracion de cada grabacion de chunk de audio


class AnalisisVozInterface(QThread):

    def __init__(self, pyaudio_ref):
        """Inicializar la clase inicializar todos los modelos y las variables de uso """
        super().__init__()
        
        print("-_-_Inicializando los modelos para STT_-_-")
        # Inicializar Pyaudio
        self.p = pyaudio_ref
        
        #
        # Modelos
        #
        self.lenguaje_analyzer = spacy.load("es_core_news_md") # Para comparar los textos
        
        modelo_huggingFace = "openai/whisper-small" # "openai/whisper-large-v3-turbo" # 
        self.parallelType = "cuda" if torch.cuda.is_available() else "cpu" # CUDA para operacion con gpu
        self.model = WhisperForConditionalGeneration.from_pretrained(modelo_huggingFace).to(self.parallelType) # Modelo para transformar audio en texto
        self.processor = WhisperProcessor.from_pretrained(modelo_huggingFace) # Procesador para _tokenizar_ el audio
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="es", task="transcribe") # Configuraciones del modelo, "en" Ingles, "es" Espanol

        self.sentimientos_analyzer = create_analyzer(task="sentiment", lang="es") # Analizador de sentimientos

        self.traductor = Translator() # Traductor de idiomas

        #
        # Variables de la clase
        #
        self.iniciado: bool = False

        self.testing = False
        self.lista_audioChunks = []

        self.emocion_actual: str = ""
        self.texto_actual: str = ""

        self.transcripcion: str = ""
        self.texto_traducido: str = ""
        self.emocion_actual: str = ""
        pass

    def run(self):
        self.proceso_main()

    def obtener_audio(self):
        """
        Retorna datos de audio en modo de array de numpy,
        el audio puede ser adquirido por medio del microfono o 
        de una fuente de un audio de archivo en caso la clase este en modo Testing
        """
        if self.testing:
            if len(self.lista_audioChunks) >= 1:
                return self.lista_audioChunks.pop(0)
            else:
                return None
        
        else:    
            '''Graba el audio del microfono por un lapso de tiempo de RECORD_SECONDS, retorna un array de numpy'''
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
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

            return audio_data
        

    def transcribir_audio(self, audio_data):
        '''Transcribe (utilizando el modelo transformer) un audio_data(array de numeros normalizados) en texto'''
        
        # Procesa el audio para que concuerde con los requerimientos del modelo
        input_features = self.processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_features.to(self.parallelType)
        
        # Generar transcripcion con "forced language decoding" (genera tokens)
        generated_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
        
        # Convierte los tokens del audio a texto
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
    

    def traducir_texto(self, texto, lang='en'):
        """Recibe un texto y lo traduce al idioma elegido (por defecto es ingles) Idiomas:
        Idiomas:
        'ar': 'arabic',
        'zh-cn': 'chinese (simplified)',
        'en': 'english',
        'fr': 'french',
        'de': 'german',
        'el': 'greek',
        'ru': 'russian',
        'pt': 'portuguese'
        """
        traduccion = self.traductor.translate(texto, src='es', dest=lang).text
        return traduccion

    
    def reconocer_emociones(self, texto):
        """
        Recibe un texto y retorna el tipo de emociones detectadas
        Retorna un texto de la forma
        (output=NEU, probas={NEU: 0.000, NEG: 0.000, POS: 0.000}
        """
        emocion = str(self.sentimientos_analyzer.predict(texto))[14:-1]
        return emocion
    
    
    def proceso_main(self) -> None:
        print("!!!!NO HA SIDO IMPLEMENTADA ESTA FUNCION: proceso_main")
        pass

    def detener_proceso_main(self):
        self.iniciado = False
        self.quit()
        self.wait()


    def test_clase(self, direccion_audio):
        """Ejecuta un test de la funcion principal de la clase, recurriendo al archivo que se entrega"""

        self.testing = True
        self.lista_audioChunks = self.cut_audio_file_in_chunks(direccion_audio)
        
        start_time = time.time() # Iniciar el contador de tiempo
        
        self.proceso_main() # La clase ya esta preparada para ejecutarse cuando es en modo testing

        end_time = time.time() # Terminar el contador de tiempo

        duracion_audio = librosa.get_duration(path=direccion_audio)
        duracion_transcripcion = end_time - start_time
        print(f"Tiempo de ejecucion de audio de {duracion_audio} segundos ->  {duracion_transcripcion:.2f} segundos")

        # Realizar la comparativa de textos
        similitud = self.comparar_texto_archivo(self.transcripcion, direccion_audio)
        print(f"Calculo de similitud: {similitud}%")


    #
    # Funciones de utilidad interna
    #
    def comparar_texto_archivo(self, texto, nombre_Archivo):
        """Compares two texts for similarity using SpaCy and returns the percentage."""

        # Cargar el texto desde el archivo
        with open(nombre_Archivo.replace('.wav', '.txt').replace('.mp3', '.txt'), 'r', encoding='utf-8') as f:
            textoOriginal = f.read()

        def preprocess_text(text):
            # Remove punctuation, special characters, and convert to lowercase
            doc = self.lenguaje_analyzer(text.lower())
            cleaned_text = " ".join([token.text for token in doc if not token.is_punct and not token.is_space])
            return cleaned_text
        
        cleaned_texto = preprocess_text(texto)
        cleaned_textoOriginal = preprocess_text(textoOriginal)

        doc1 = self.lenguaje_analyzer(cleaned_texto)
        print(doc1.text)

        doc2 = self.lenguaje_analyzer(cleaned_textoOriginal)
        print(doc2.text)

        similarity_score = doc1.similarity(doc2)

        return similarity_score * 100  # Convertir a porcentaje


    def cut_audio_file_in_chunks(self, filePath):
        """" Corta un archivo de audio en chunks para que sea transcrito por partes """

        # Cortar el archivo de audio en bloques
        audio_array, _ = librosa.load(filePath, sr=RATE) # librosa se encarga de cargar archivos de audio 

        # Calcular el numero de samples per chunk
        chunk_samples = RATE * RECORD_SECONDS

        # Split el audio en chunks y transcribir cada uno
        num_chunks = int(np.ceil(len(audio_array) / chunk_samples))
        array_audios = []

        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i+1) * chunk_samples, len(audio_array))

            # Extraer el chunk
            audio_chunk = audio_array[start_sample:end_sample]
            
            # rellenar con ceros (para el ultimo chunk)
            if len(audio_chunk) < chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)), mode="constant")
            
            # Agregar el chink
            array_audios.append(audio_chunk)

        return array_audios

        