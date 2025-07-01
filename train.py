from torch.utils.data import Dataset
import os
import pandas as pd
from pathlib import Path
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T


train_transformer = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025
    ),
    T.AmplitudeToDB(),
    T.FrequencyMasking(freq_mask_param=30),
    T.TimeMasking(time_mask_param=80)
) 

val_transformer = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025
    ),
    T.AmplitudeToDB()
) 

class ESC50Dataset(Dataset):
    def __init__(self, data_directory, metadata_file, split="train", transformer=None):
        super.__init__()
        self.data_dir = Path(data_directory)
        self.metadata = pd.read_csv(metadata_file)
        self.transformer = transformer
        self.split = split

        if self.split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]
        
        self.classes = sorted(self.metadata["category"].unique())
        self.classes_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.classes_to_idx)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.data_dir, "audio", row["filename"])
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.transformer:
            spectrogram = self.transformer(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']

