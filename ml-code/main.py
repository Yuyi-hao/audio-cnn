import torchaudio.transforms as T
import torch.nn as nn
import torch

class AudioProcessor:
    def __init__(self):
        self.transformer = nn.Sequential(
            T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025),
            T.AmplitudeToDB(),
        )
    
    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transformer(waveform)
        return spectrogram.unsqueeze(0)

class AudioClassifier:
    def load_model(self):
        pass