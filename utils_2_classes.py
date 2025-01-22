# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
import torch
import torchaudio
import numpy as np
from glob import glob
from torch.utils.data import Dataset

# 2-class label map
label_dict = {
    "heartbeat": 0,
    "non_heartbeat": 1
}

SR = 16000  # sample rate for audio

#############################
#        DATASET
#############################

class TwoClassDataset(Dataset):
    """
    Loads data from a directory with subfolders:
       data_dir/
         heartbeat/
         non_heartbeat/
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []
        self.labels = []

        for class_name in label_dict.keys():
            folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(folder):
                # skip if the sub-folder does not exist
                continue
            # gather .wav files
            wav_paths = glob(os.path.join(folder, "*.wav"))
            for wpath in wav_paths:
                self.data_list.append(wpath)
                self.labels.append(label_dict[class_name])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        sample, sr = torchaudio.load(audio_path)
        # optionally resample if your audio is not in 16k
        if sr != SR:
            resample = torchaudio.transforms.Resample(sr, SR)
            sample = resample(sample)

        if self.transform:
            sample = self.transform(sample)

        label = self.labels[idx]
        return sample, label

#############################
#       AUDIO AUG/FEATS
#############################

def spec_augment(
    x,
    frequency_masking_para=20,
    time_masking_para=20,
    frequency_mask_num=2,
    time_mask_num=2
):
    """
    Simple SpecAugment on log-mel:
        x: shape [1, n_mels, time_frames]
    """
    _, lenF, lenT = x.shape
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.randint(0, frequency_masking_para + 1)
        f0 = random.randint(0, max(0, lenF - f))
        x[:, f0 : f0 + f, :] = 0

    # Time masking
    for _ in range(time_mask_num):
        t = np.random.randint(0, time_masking_para + 1)
        t0 = random.randint(0, max(0, lenT - t))
        x[:, :, t0 : t0 + t] = 0

    return x


class LogMel:
    def __init__(
        self,
        device,
        sample_rate=SR,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        ).to(device)
        self.device = device

    def __call__(self, x):
        # x: shape [B, 1, T] or [1, T]
        mel_out = self.mel(x)
        return (mel_out + 1e-6).log()


class Padding:
    """
    Zero-pad (or fail) so that each sample is 1s of audio at 16kHz
    """
    def __init__(self, target_len=SR):
        self.target_len = target_len

    def __call__(self, x):
        # x: shape [1, T]
        pad_len = self.target_len - x.shape[-1]
        if pad_len > 0:
            # zero-pad to the right
            x = torch.cat([x, torch.zeros([1, pad_len])], dim=-1)
        elif pad_len < 0:
            # or slice if longer
            x = x[:, : self.target_len]
        return x


class Preprocess:
    """
    1) Optionally add noise
    2) Time shift
    3) Convert to log-mel
    4) (Optional) SpecAugment
    """

    def __init__(
        self,
        noise_loc=None,
        device="cpu",
        sample_rate=SR,
        specaug=False,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        self.device = device
        self.specaug = specaug
        self.sample_rate = sample_rate
        self.feature = LogMel(device, sample_rate=sample_rate)
        self.background_noise = []
        if noise_loc is not None and os.path.isdir(noise_loc):
            # load all noise wav files
            noise_files = glob(os.path.join(noise_loc, "*.wav"))
            for nf in noise_files:
                audio, sr = torchaudio.load(nf)
                if sr != sample_rate:
                    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
                self.background_noise.append(audio.to(device))

        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.frequency_mask_num = frequency_mask_num
        self.time_mask_num = time_mask_num

    def __call__(self, x, labels, augment=True, is_train=True, noise_prob=0.5):
        """
        x: shape [B, 1, T]
        labels: shape [B]
        """
        B = x.shape[0]
        # optional noise injection & time shift
        if augment and is_train:
            for i in range(B):
                # Maybe add noise
                if len(self.background_noise) > 0 and random.random() < noise_prob:
                    noise = random.choice(self.background_noise)
                    # pick random snippet from noise if longer than 1s
                    if noise.shape[-1] >= x.shape[-1]:
                        start = random.randint(0, noise.shape[-1] - x.shape[-1])
                        noise_slice = noise[:, start : start + x.shape[-1]]
                    else:
                        # noise is shorter, just reuse it
                        noise_slice = noise
                    amp = np.random.uniform(0, 0.2)
                    x[i] = torch.clamp(x[i] + amp * noise_slice, -1.0, 1.0)

                # time shift
                shift = int(np.random.uniform(-0.1, 0.1) * self.sample_rate)
                if shift != 0:
                    if shift > 0:
                        # shift to right
                        x[i] = torch.cat(
                            [torch.zeros((1, shift), device=self.device), x[i][:, :-shift]],
                            dim=-1,
                        )
                    else:
                        # shift to left
                        shift = abs(shift)
                        x[i] = torch.cat(
                            [x[i][:, shift:], torch.zeros((1, shift), device=self.device)],
                            dim=-1,
                        )

        # Convert to log-mel
        x = self.feature(x)

        # SpecAugment if desired
        if self.specaug and augment and is_train:
            for i in range(B):
                x[i] = spec_augment(
                    x[i],
                    frequency_masking_para=self.frequency_masking_para,
                    time_masking_para=self.time_masking_para,
                    frequency_mask_num=self.frequency_mask_num,
                    time_mask_num=self.time_mask_num,
                )
        return x
