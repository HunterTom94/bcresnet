import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from bcresnet import BCResNets
from utils_2_classes import Padding, Preprocess

def load_true_intervals(label_file):
    intervals = []
    if not os.path.isfile(label_file):
        return intervals

    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = parts
                if label.lower() == 'b':
                    start_sec = float(start)
                    end_sec = float(end)
                    intervals.append((start_sec, end_sec, i + 1))
    return intervals

def overlaps_any_true_region(start_s, end_s, true_intervals, overlap_threshold=0.3):
    for (region_start, region_end, region_idx) in true_intervals:
        region_duration = region_end - region_start
        overlap_start = max(start_s, region_start)
        overlap_end = min(end_s, region_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration >= (overlap_threshold * region_duration):
            return True

    return False

def get_true_windows_from_generation_logic(
    duration_s,
    sr,
    true_intervals,
    window_size_s=0.200,
    step_size_s=0.050,
    overlap_threshold=0.3
):
    generated_true_windows = []
    start_s = 0.0

    while True:
        end_s = start_s + window_size_s
        if end_s > duration_s:
            break

        if overlaps_any_true_region(start_s, end_s, true_intervals, overlap_threshold):
            generated_true_windows.append((start_s, end_s))

        start_s += step_size_s

    return generated_true_windows

class AudioInference:
    def __init__(
            self,
            model_path,
            tau=8,
            device=None,
            noise_dir="./data/speech_commands_v0.02/_background_noise_",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = BCResNets(int(tau * 8), num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.sample_rate = 16000
        self.padding_transform = Padding()
        self.preprocess = Preprocess(
            noise_dir,
            self.device,
            specaug=False,
            frequency_masking_para=0
        )
        self.index_to_label = [
            "beat",
            "non-beat"
        ]

    def infer_window(self, chunk_waveform):
        chunk_waveform = chunk_waveform.unsqueeze(0)

        if chunk_waveform.shape[-1] > self.sample_rate:
            chunk_waveform = chunk_waveform[..., :self.sample_rate]

        chunk_waveform = self.padding_transform(chunk_waveform)
        chunk_waveform = chunk_waveform.to(self.device)

        chunk_waveform = self.preprocess(
            chunk_waveform,
            labels=None,
            is_train=False,
            augment=False
        )

        with torch.no_grad():
            logits = self.model(chunk_waveform)
            pred_idx = torch.argmax(logits, dim=-1).item()

        pred_label = self.index_to_label[pred_idx]
        return pred_idx, pred_label

    def __call__(
        self,
        wav_path,
        window_ms=500,
        step_ms=250,
        snippet_window_s=0.200,
        snippet_step_s=0.050,
    ):
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = waveform.cpu()
        n_samples = waveform.shape[-1]

        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        true_intervals = load_true_intervals(txt_path)

        window_size = int((window_ms / 1000.0) * sr)
        step_size = int((step_ms / 1000.0) * sr)

        predicted_windows = []
        start = 0
        while True:
            end = start + window_size
            if start >= n_samples:
                break

            chunk = waveform[..., start:end]
            pred_idx, pred_label = self.infer_window(chunk)

            start_sec = start / sr
            end_sec = min(end, n_samples) / sr
            predicted_windows.append((start_sec, end_sec, pred_idx, pred_label))

            start += step_size
            if start >= n_samples:
                break

        waveform_np = waveform.squeeze(0).numpy()
        time_axis = np.arange(n_samples) / float(sr)
        duration_s = n_samples / sr

        generated_true_windows = get_true_windows_from_generation_logic(
            duration_s=duration_s,
            sr=sr,
            true_intervals=true_intervals,
            window_size_s=snippet_window_s,
            step_size_s=snippet_step_s,
            overlap_threshold=0.3
        )

        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

        axs[0, 0].plot(time_axis, waveform_np, label="Audio waveform", color="blue")
        is_first_gt = True
        for (st_sec, ed_sec, region_idx) in true_intervals:
            if is_first_gt:
                axs[0, 0].axvspan(st_sec, ed_sec, color="green", alpha=0.2, label="TRUE region (from file)")
                is_first_gt = False
            else:
                axs[0, 0].axvspan(st_sec, ed_sec, color="green", alpha=0.2)

        axs[0, 0].set_title("TRUE region from file")
        axs[0, 0].set_ylabel("Amplitude")
        axs[0, 0].legend()

        axs[0, 1].plot(time_axis, waveform_np, label="Audio waveform", color="blue")
        is_first_gt = True
        for (st_sec, ed_sec) in generated_true_windows:
            if is_first_gt:
                axs[0, 1].axvspan(st_sec, ed_sec, color="green", alpha=0.2, label="TRUE region (generated logic)")
                is_first_gt = False
            else:
                axs[0, 1].axvspan(st_sec, ed_sec, color="green", alpha=0.2)

        axs[0, 1].set_title("TRUE region from generated logic")
        axs[0, 1].set_ylabel("Amplitude")
        axs[0, 1].legend()

        for ax in axs[1, :]:
            ax.plot(time_axis, np.zeros_like(waveform_np), color="white", alpha=0.0)
            is_first_pred = True
            for (st, ed, pidx, plabel) in predicted_windows:
                if pidx == 0:
                    if is_first_pred:
                        ax.axvspan(st, ed, color="red", alpha=0.3, label="Predicted = beat")
                        is_first_pred = False
                    else:
                        ax.axvspan(st, ed, color="red", alpha=0.3)

            ax.set_ylabel("Prediction")
            ax.set_ylim(-0.1, 1.0)
            ax.legend()

        axs[1, 0].set_title("Predicted beats for file TRUE regions")
        axs[1, 1].set_title("Predicted beats for generated TRUE regions")

        for ax in axs[1, :]:
            ax.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model_path = "./model_2_classes.pth"

    wav_path = "G:\\CodeRepo\\k-sound-data\\track_label\\audio_output_20240525_202854_530390_6.wav"
    # wav_path = "G:\\CodeRepo\\k-sound-data\\track_label\\audio_output_20240613_002635_920090_17.wav"

    inference_engine = AudioInference(model_path=model_path, tau=8)

    inference_engine(
        wav_path=wav_path,
        window_ms=200,
        step_ms=50
    )
