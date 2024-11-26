import concurrent
import queue
import random
import threading
import time

import mpipe
import numpy as np
from AnalisisVozInterfaz import AnalisisVozInterface

import pyaudio

import os

class AnalisisVoz_Dummy(AnalisisVozInterface):

    def proceso_main(self):

        self.iniciado = True
        emociones_random = ["NEU", "NEG", "POS"]
        palabras_random = ["perfil", "paquete", "capa", "tramposo", "castillo", "skibidi", "preferencia", "seguir", "gato", "CUANTICO", "coro"]
        rate = 0
        largo = 0
        traduccion = ""
        while self.iniciado:
            rate = random.randint(1,4)
            largo = random.randint(3,6)
            time.sleep(rate)

            self.texto_actual = ""
            for i in range(largo):
                traduccion = traduccion + random.choice(palabras_random) + " "
            
            self.texto_actual = traduccion
            self.texto_traducido = traduccion[::-1]
            self.emocionActual = random.choice(emociones_random)

            self.texto_Actual_Compartido = self.texto_actual

class AnalisisVoz_Secuencial(AnalisisVozInterface):

    def proceso_main(self):
        """ Funcion para iniciar la transcripcion usando el microfono como constante entrada de audio """
        self.iniciado = True
        try:
            while self.iniciado:
                audio_data = self.obtener_audio()
                if audio_data is not None:

                    """Se hace un analisis del audio_data ingresado, insertando valores en self.texto_actual, self.texto_traducido y self.emocion_actual"""
                    # Transcribir audio con el modelo
                    print("Transcripcion:", end='')
                    transcription_actual = self.transcribir_audio(audio_data)
                    self.texto_actual = transcription_actual
                    self.transcripcion += transcription_actual
                    print(self.texto_actual)

                    # Traducir el texto
                    print("| Traduccion Ingles: ", end='')
                    self.texto_traducido = self.traducir_texto(self.texto_actual)
                    self.transcripcion_traducido += self.texto_traducido
                    print(self.texto_traducido)

                    # Analizar el sentimiento
                    print("| Reconocimiento de emociones: ", end='')
                    self.emocion_actual = self.reconocer_emociones(self.texto_actual)
                    print(self.emocion_actual)

                else:
                    self.iniciado = False
        
        except KeyboardInterrupt:
            print("Deteniendo transcripcion desde microfono.")

class AnalisisVoz_Paralelo_ConcurrentFeatures(AnalisisVozInterface):

    def proceso_main(self):
        """ Funcion para iniciar la transcripcion usando el microfono como constante entrada de audio """
        
        self.iniciado = True
        try:
            while self.iniciado:
                audio_data = self.obtener_audio()
                if audio_data is not None:
                    
                    print("Transcripcion:", end='')
                    transcription = self.transcribir_audio(audio_data)
                    self.texto_actual = transcription
                    print(self.texto_actual)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Ejecutar las funciones en paralelo y obtener sus resultados, se ejecutan en hilos separados
                        results = list(executor.map(lambda func: func(self.texto_actual), [self.traducir_texto, self.reconocer_emociones]))
                    
                    self.texto_traducido = results[0]
                    self.emocion_actual = results[1]
                    
                    print("| Traduccion de texto: ", end='')
                    print(self.texto_traducido)
                    print("| Reconocimiento de emociones: ", end='')
                    print(self.emocion_actual)
                    pass

                else:
                    self.iniciado = False
        
        except KeyboardInterrupt:
            print("Deteniendo transcripcion desde microfono.")

class AnalisisVoz_Paralelo_Features(AnalisisVozInterface):

    def __init__(self, pyaudio_ref):
        super().__init__(pyaudio_ref)
        self.iniciado = False
        self.lock = threading.Lock()  # Para sincronizar el acceso a las variables compartidas
    
    def proceso_main(self):
        """Funcion para iniciar la transcripción usando el microfono como constante entrada de audio"""
        
        self.iniciado = True
        try:
            while self.iniciado:
                audio_data = self.obtener_audio()  # Obtienes los datos de audio (chunk)
                
                if audio_data is not None:
                    # Transcripción en el hilo principal
                    transcription = self.transcribir_audio(audio_data)
                    self.texto_actual = transcription
                    
                    # Lanzamos las tareas de traducción y análisis de emociones en paralelo
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Ejecutar funciones en paralelo (traducción y emoción) sin bloquear la transcripción
                        future_traduccion = executor.submit(self.traducir_texto, self.texto_actual)
                        future_emocion = executor.submit(self.reconocer_emociones, self.texto_actual)
                        
                        # Esperamos las respuestas de ambas tareas sin bloquear la grabación
                        self.texto_traducido = future_traduccion.result()
                        self.emocion_actual = future_emocion.result()

                    # Imprimir los resultados de manera continua sin bloquear la grabación
                    self.mostrar_resultados()
                
                else:
                    self.iniciado = False

        except KeyboardInterrupt:
            print("Deteniendo transcripción desde el micrófono.")

class AnalisisVoz_Paralelo_ProCons_Threads(AnalisisVozInterface):

    def __init__(self, pyaudio_ref):
        super().__init__(pyaudio_ref)

        self.iniciado = False
        self.audio_queue = queue.Queue()

    def work_obtener_audio(self):
        while self.iniciado:
            data = self.obtener_audio()
            self.audio_queue.put(data)

    def procesar_pipeline(self):
        while self.iniciado:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1)

                    transcription_actual = self.transcribir_audio(audio_data)
                    self.texto_actual = transcription_actual
                    self.transcripcion += transcription_actual
                    # Traducir el texto
                    self.texto_traducido = self.traducir_texto(self.texto_actual)
                    self.transcripcion_traducido += self.texto_traducido

                    # Analizar el sentimiento
                    self.emocion_actual = self.reconocer_emociones(self.texto_actual)

                    print(f"Transcripción: {self.texto_actual}")
                    print(f"Traducción: {self.texto_traducido}")
                    print(f"Emoción: {self.emocion_actual}")
                    
                else:
                    continue

            except queue.Empty:
                print("Lista vacia, pasando al siguiente")
                #No pasa nada de esar vacia
                continue

    def proceso_main(self):
        """Funcion para iniciar la transcripción usando el microfono como constante entrada de audio"""
        self.iniciado = True

        hilo_audio = threading.Thread(target=self.work_obtener_audio)
        hilo_audio.daemon = True

        hilo_pipeline = threading.Thread(target=self.procesar_pipeline)
        hilo_pipeline.daemon = True

        # Iniciar cada hilo
        hilo_audio.start()
        hilo_pipeline.start()
        
        try:
            while self.iniciado:
                continue
        except KeyboardInterrupt:
            print("Deteniendo transcripción desde el micrófono.")
            self.iniciado = False
            hilo_audio.join()
            hilo_pipeline.join()


# Funcion Main
if __name__ == "__main__":

    p_audio = pyaudio.PyAudio()

    TranscripcionManager = AnalisisVoz_Secuencial(p_audio)

    audio_file_path = 'audiosTest/'

    #TranscripcionManager.analizar_desde_microfono()
    '''
    audio_files = [os.path.join(audio_file_path, f) for f in os.listdir(audio_file_path) if f.endswith(('.wav', '.mp3'))]
    for file_path in audio_files:
        TranscripcionManager.test_clase(file_path)
    '''
    TranscripcionManager.proceso_main()