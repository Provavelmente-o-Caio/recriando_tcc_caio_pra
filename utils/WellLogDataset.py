import numpy as np
import torch
from torch.utils.data import Dataset


class WellLogDataset(Dataset):
    def __init__(
        self, features_data, target_data, sequence_length, mask_value, augmentation=None
    ):
        self.features_data = features_data
        self.target_data = target_data
        self.sequence_length = sequence_length
        self.mask_value = mask_value
        self.augmentation = augmentation
        self.sequences = []
        self.targets = []
        self.training = True
        self._create_sequences()

    def _create_sequences(self):
        for i in range(len(self.features_data) - self.sequence_length):
            if i + self.sequence_length < len(self.target_data):
                sequence_features = self.features_data[i : i + self.sequence_length]
                target_value = self.target_data[i + self.sequence_length]
                if target_value != self.mask_value:
                    self.sequences.append(sequence_features)
                    self.targets.append(target_value)

    def train(self):
        """Ativa augmentação (modo treino)"""
        self.training = True
        return self

    def eval(self):
        """Desativa augmentação (modo inferência)"""
        self.training = False
        return self

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = np.array(self.sequences[index])  # Convert to a numpy array

        # Apply augmentation if specified
        if self.augmentation and self.training:
            sequence = self.augmentation(sequence)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32).unsqueeze(-1)
        return sequence, target


class WellLogAugmentation:
    def __init__(self, noise_level=0.01, scale_range=(0.95, 1.05)):
        self.noise_level = noise_level
        self.scale_range = scale_range

    def __call__(self, sequence):
        if isinstance(sequence, list):
            sequence = np.array(sequence)

        # Add Gaussian noise
        if np.random.rand() < 0.5:  # 50% chance to apply noise
            noise = np.random.normal(0, self.noise_level, sequence.shape)
            sequence = sequence + noise

        # Randomly scale the sequence
        if np.random.rand() < 0.5:  # 50% chance to apply scaling
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            sequence = sequence * scale_factor

        return sequence
