from abc import ABC, abstractmethod
import random
import time
from PyQt5.QtCore import QThread, pyqtSignal

palabras_random = [
    "perfil",
    "paquete",
    "capa",
    "tramposo",
    "castillo",
    "skibidi",
    "preferencia",
    "seguir",
    "gato",
    "CUANTICO",
    "coro",
    "gobernador",
    "mejorar",
    "maíz",
    "táctica",
    "aniversario",
    "nativo",
    "EL PEPE",
    "órbita",
    "autorizar",
    "borrar",
    "dulce",
    "crouch",
    "novato",
]

emociones_random = ["Positivo", "Neutral", "Negativo"]

#No puedo heradar dos clases al mismo tiempo xddd
class AnalisisVoz(QThread):

    def __init__(self):
        super().__init__()
        #print("Hilo iniciado.")
        self.emocionActual: str = ""
        self.texto_actual: str = ""
        self.texto_traducido: str = ""
        self.transcripcion = ""

        self.iniciado: bool = False

    def run(self):
        self.analizar_desde_microfono()

    def analizar_desde_microfono(self):
        try:
            print("Hilo iniciado.")
            self.iniciado = True
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
                #print(self.texto_actual)

        except Exception as e:
            print(f"Error en el hilo: {e}")
        

    
    def detener_transcribir_desde_microfono(self):
        self.iniciado = False

        self.quit()  # Solicita detener el hilo
        self.wait() #Para esperar a que el hilo termine su tarea

