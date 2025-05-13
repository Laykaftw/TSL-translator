import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.data = []
        self.labels = []
        self.label_map = {}
        
        # Dynamically assign labels
        for sign_idx, sign_dir in enumerate(os.listdir(data_dir)):
            self.label_map[sign_dir] = sign_idx
            sign_path = os.path.join(data_dir, sign_dir)
            
            for video_dir in os.listdir(sign_path):
                video_path = os.path.join(sign_path, video_dir, "preprocessed")
                
                if os.path.exists(video_path):
                    frame_files = sorted(
                        [f for f in os.listdir(video_path) if f.startswith("frame")],
                        key=lambda x: int(x.split("_")[1].split(".")[0])
                    )
                    
                    # Load sequences of frames
                    for i in range(len(frame_files) - sequence_length + 1):
                        seq_files = frame_files[i:i+sequence_length]
                        seq = [np.load(os.path.join(video_path, f)) for f in seq_files]
                        self.data.append(np.stack(seq))
                        self.labels.append(sign_idx)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]