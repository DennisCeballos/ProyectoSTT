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