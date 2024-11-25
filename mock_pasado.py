from abc import ABC, abstractmethod
import random
import time

from AnalisisVozInterfaz import AnalisisVozInterface

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



class AnalisisVoz(AnalisisVozInterface):

    def __init__(self):
        self.emocionActual: str = ""
        self.texto_actual: str = ""
        self.texto_traducido: str = ""

        self.iniciado: bool = True
        pass

    def analizar_desde_microfono(self):
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
            print(self.texto_actual)
    
    def detener_transcribir_desde_microfono(self):
        self.iniciado = False

e = AnalisisVoz()
e.analizar_desde_microfono()