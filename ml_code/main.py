import torchaudio.transforms as T
import torch.nn as nn
import torch
import os
import soundfile as sf
import io
import base64
import numpy as np
import librosa
from .model import AudioCNN

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        audio_cnn_model = torch.load('./models/best_model.pth')
        self.classes = audio_cnn_model['classes']


def predict_audio_from_model(audio_b64, device='cpu'):
    # mimicking web request
    # preload objects
    audio_processor = AudioProcessor()
    audio_cnn_model = torch.load('./ml_code/models/best_model.pth')
    classes = audio_cnn_model['classes']

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(audio_cnn_model['model_state_dict'])
    model.to(device)
    model.eval()

    audio_bytes = base64.b64decode(audio_b64)

    audio_data, sample_rate = sf.read(
        io.BytesIO(audio_bytes), dtype="float32")

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != 44100:
        audio_data = librosa.resample(
            y=audio_data, orig_sr=sample_rate, target_sr=44100)
    

    spectrogram = audio_processor.process_audio_chunk(audio_data)
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        output, feature_maps = model(
            spectrogram, return_feature_maps=True)

        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

        predictions = [{"class": classes[idx.item()], "confidence": prob.item()}
                        for prob, idx in zip(top3_probs, top3_indicies)]

        viz_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:  # [batch_size, channels, height, width]
                aggregated_tensor = torch.mean(tensor, dim=1)
                squeezed_tensor = aggregated_tensor.squeeze(0)
                numpy_array = squeezed_tensor.cpu().numpy()
                clean_array = np.nan_to_num(numpy_array)
                viz_data[name] = {
                    "shape": list(clean_array.shape),
                    "values": clean_array.tolist()
                }

        spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
        clean_spectrogram = np.nan_to_num(spectrogram_np)

        max_samples = 8000
        waveform_sample_rate = 44100
        if len(audio_data) > max_samples:
            step = len(audio_data) // max_samples
            waveform_data = audio_data[::step]
        else:
            waveform_data = audio_data

    response = {
        "predictions": predictions,
        "visualization": viz_data,
        "input_spectrogram": {
            "shape": list(clean_spectrogram.shape),
            "values": clean_spectrogram.tolist()
        },
        "waveform": {
            "values": waveform_data.tolist(),
            "sample_rate": waveform_sample_rate,
            "duration": len(audio_data) / waveform_sample_rate
        }
    }

    return response


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_data, sample_rate = sf.read('ESC-50-dataset/audio/1-15689-A-4.wav')
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    print(audio_b64[:10])
    return
    result = predict_audio_from_model(audio_b64, device)
    print(result["predictions"])

if __name__=="__main__":
    main()