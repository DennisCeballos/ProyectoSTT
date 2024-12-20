from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QTextEdit
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap

from audioPro import *

from microphone_manager import MicrophoneManager
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.abspath("./resources/ui/main.ui")
        uic.loadUi(ui_path,self)

        self.microphone_manager = MicrophoneManager()
        self.populate_microphone_list()

        #
        # ACA CAMBIAR LA CLASE QUE SE ENCARGARA DE MANEJAR EL STT
        #
        self.analizador_voz: AnalisisVozInterface = AnalisisVoz_Secuencial(self.microphone_manager.audio)

        # Timer para actualizar la pantalla constantemente
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(500)  # 500 ms = 0.5 segundos
        self.update_timer.timeout.connect(self.update_ui)

        self.bt_grabar.clicked.connect(self.on_bt_grabar_clicked)
        self.bt_stop.clicked.connect(self.on_bt_stop_clicked)
        self.cb_inputList.currentIndexChanged.connect(self.on_microphone_selected)
        self.cb_language.currentIndexChanged.connect(self.on_microphone_selected)

        # Configurar imagenes de emociones
        self.pixmap_positivo = QPixmap('./resources/images/positivo.png')
        self.pixmap_neutral = QPixmap('./resources/images/neutral.png')
        self.pixmap_negativo = QPixmap('./resources/images/negativo.png')
        self.pixmap_unknown = QPixmap('./resources/images/error404.png')
        self.img_status = self.findChild(QLabel, "img_status")


        self.show()

    def update_ui(self):
        self.tf_textoTraducido.setText(self.analizador_voz.texto_traducido)
        self.tf_textoTranscrito.setText(self.analizador_voz.transcripcion)
        emocion = self.analizador_voz.emocion_actual
        self.label_4.setText(emocion)
        if emocion[:3] == "NEU":
            self.img_status.setPixmap(self.pixmap_neutral)
        elif emocion[:3] == "NEG":
            self.img_status.setPixmap(self.pixmap_negativo)
        elif emocion[:3] == "POS":
            self.img_status.setPixmap(self.pixmap_positivo)
        else:
            self.img_status.setPixmap(self.pixmap_unknown)

    # Eventos
    def on_bt_grabar_clicked(self):
        self.lb_status.setText("Grabando...")
        self.cb_inputList.setEnabled(False)
        self.cb_language.setEnabled(False)
        
        if not self.analizador_voz.isRunning():
            self.analizador_voz.start()

        # Iniciar timer que actualiza la pantalla
        self.update_timer.start()
            

    def on_bt_stop_clicked(self):
        self.lb_status.setText("Parado")
        self.cb_inputList.setEnabled(True)
        self.cb_language.setEnabled(False)
        
        # Detener el análisis en segundo plano
        if self.analizador_voz.isRunning():
            self.analizador_voz.detener_proceso_main()

        # Detener el timer que actualiza la pantalla
        self.update_timer.stop()
    

    def on_microphone_selected(self):
        selected_index = self.cb_inputList.currentData()
        if self.microphone_manager.select_microphone(selected_index):
            selected_microphone = self.microphone_manager.get_selected_microphone()
            print(f"Micrófono seleccionado: {selected_microphone}")
        else:
            print("Error al seleccionar el micrófono.")

    # funciones
    def populate_microphone_list(self):
        microphones = self.microphone_manager.get_microphone_list()
        self.cb_inputList.clear()
        for name, index in microphones:
            self.cb_inputList.addItem(name, index)


def initialize_window(argv: list[str]):
    app = QApplication(argv)
    UIWindow = MainWindow()
    UIWindow.setWindowTitle('Super Transcriptor Paralelo 64');
    app.exec_()
    