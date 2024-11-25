import pyaudio

class MicrophoneManager:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.devices = []
        self.selected_device_index = None

    def get_microphone_list(self):
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                name = device_info.get('name')
                try:
                    name = name.encode('latin1').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

                self.devices.append((name, i))
        return self.devices

    def select_microphone(self, device_index):
        for _, index in self.devices:
            if device_index == index:
                self.selected_device_index = device_index
                return True
        return False

    def get_selected_microphone(self):
        if self.selected_device_index is not None:
            for name, index in self.devices:
                if index == self.selected_device_index:
                    return (name, index)
        return None

    def terminate(self):
        self.audio.terminate()
