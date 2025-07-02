from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

os.makedirs('./models', exist_ok=True)

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
        super().__init__()
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

def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam*x+(1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# data preparation
es50_dataset = "ESC-50-dataset"

train_dataset = ESC50Dataset(data_directory=es50_dataset, metadata_file=os.path.join(es50_dataset, "meta", "esc50.csv"), split="train", transformer=train_transformer)
val_dataset = ESC50Dataset(data_directory=es50_dataset, metadata_file=os.path.join(es50_dataset, "meta", "esc50.csv"), split="val", transformer=train_transformer)

print(f"Training data samples: {len(train_dataset)}")
print(f"Validation data samples: {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# model and device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audioCNN_model = AudioCNN(num_classes=len(train_dataset.classes))
optimizer = optim.AdamW(audioCNN_model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
audioCNN_model.to(device)

num_epochs = 100

schedular = OneCycleLR(
    optimizer,
    max_lr=0.002,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.1
)



# training loop
best_accuracy = 0.0
for epoch in range(num_epochs):
    audioCNN_model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        if np.random.random() > 0.7:
            data, target_a, target_b, lam = mixup_data(data, target)
            output = audioCNN_model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = audioCNN_model(data)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}"
        })
    avg_epoch_loss = epoch_loss/len(train_dataloader)
    audioCNN_model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = audioCNN_model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted==target).sum().item()
    
    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_dataloader)

    print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'model_state_dict': audioCNN_model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
            'classes': train_dataset.classes
        }, './models/best_model.pth')
        print(f'New best model saved: {accuracy:.2f}%')

print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')