import concurrent
import threading

import mpipe
from AnalisisVozInterfaz import AnalisisVozInterface

import pyaudio

import os

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

    def mostrar_resultados(self):
        """Muestra los resultados de transcripción, traducción y emoción"""
        print(f"Transcripción: {self.texto_actual}")
        print(f"Traducción: {self.texto_traducido}")
        print(f"Emoción: {self.emocion_actual}")

class AnalisisVoz_Paralelo_MPipe(AnalisisVozInterface):

    def __init__(self, pyaudio_ref):
        super().__init__(pyaudio_ref)
        self.iniciado = False
        self.lock = threading.Lock()  # Para sincronizar el acceso a las variables compartidas
    
    def proceso_main(self):
        """Funcion para iniciar la transcripción usando el microfono como constante entrada de audio"""
        
        
        self.iniciado = True
        stage1 = mpipe.OrderedStage(self.obtener_audio)
        stage2 = mpipe.OrderedStage(self.transcribir_audio)
        stage3 = mpipe.OrderedStage(self.traducir_texto)
        stage4 = mpipe.OrderedStage(self.reconocer_emociones)
        stage1.link(stage2)
        stage2.link(stage3)
        stage2.link(stage4)
        pipe = mpipe.Pipeline(stage1)
        
        try:
            while self.iniciado:
                audio_data = self.obtener_audio()  # Obtienes los datos de audio (chunk)
                
                if audio_data is not None:
                    pipe.put(audio_data)
                    pipe.put(None)
                    
                    result = pipe.get()
                    print(result)
                    
                else:
                    self.iniciado = False

        except KeyboardInterrupt:
            print("Deteniendo transcripción desde el micrófono.")

    def mostrar_resultados(self):
        """Muestra los resultados de transcripción, traducción y emoción"""
        print(f"Transcripción: {self.texto_actual}")
        print(f"Traducción: {self.texto_traducido}")
        print(f"Emoción: {self.emocion_actual}")


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