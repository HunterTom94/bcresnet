import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from bcresnet import BCResNets
from utils import Padding, Preprocess


class AudioInference:
    def __init__(
            self,
            model_path,
            tau=8,
            device=None,
            noise_dir="./data/speech_commands_v0.02/_background_noise_",
    ):
        """
        Initialize the inference class.

        :param model_path: Path to the saved .pth file
        :param tau: Must match the tau used during training (default=8)
        :param device: 'cuda' or 'cpu'; if None, will auto-detect
        :param noise_dir: Path to background noise folder,
                          should match the directory used in training.
        """
        # 1) Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2) Load model
        self.model = BCResNets(int(tau * 8)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 3) Audio settings
        self.sample_rate = 16000  # Google Speech Commands is typically 16k

        # 4) Same zero-padding transform as training
        self.padding_transform = Padding()

        # 5) Same test-time Preprocess (no frequency masking, no random augment)
        self.preprocess = Preprocess(
            noise_dir,
            self.device,
            specaug=False,
            frequency_masking_para=0
        )

        # 6) Mapping from class index to label name (as used in your training)
        self.index_to_label = [
            "yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go", "unknown", "silence"
        ]

    def infer_window(self, chunk_waveform):
        """
        Inference on a single chunk (e.g., 500ms).
        Model expects 1s input (16,000 samples), so we will
        zero-pad from chunk length to 16,000. Then run a forward pass.

        :param chunk_waveform: shape (1, time) in mono
        :return: predicted index (int), predicted label (str)
        """
        # 1) shape => (batch=1, 1, time)
        chunk_waveform = chunk_waveform.unsqueeze(0)  # (1,1,chunk_len)

        # 2) Ensure length up to 16000:
        #    - If chunk is longer than 1s, we clip
        #    - If it's shorter, we pad
        if chunk_waveform.shape[-1] > self.sample_rate:
            chunk_waveform = chunk_waveform[..., : self.sample_rate]

        # 3) Zero-pad if under 16000 samples
        chunk_waveform = self.padding_transform(chunk_waveform)

        # 4) Send to device
        chunk_waveform = chunk_waveform.to(self.device)

        # 5) Preprocess (no label, is_train=False, no augment)
        chunk_waveform = self.preprocess(chunk_waveform, labels=None, is_train=False, augment=False)

        # 6) Forward pass
        with torch.no_grad():
            logits = self.model(chunk_waveform)
            pred_idx = torch.argmax(logits, dim=-1).item()

        pred_label = self.index_to_label[pred_idx]
        return pred_idx, pred_label

    def __call__(self, wav_path, window_ms=500, step_ms=250):
        """
        Run sliding-window inference on an audio of arbitrary length.

        :param wav_path: Path to audio file
        :param window_ms: Window size in milliseconds (default=500ms)
        :param step_ms: Step size in milliseconds (default=250ms)
        """
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # ----------------------------------------------------------------------
        # 1) Load the entire audio, resample, convert to mono
        # ----------------------------------------------------------------------
        waveform, sr = torchaudio.load(wav_path)

        # If not 16kHz, resample
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        # If stereo, take only first channel
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        # waveform shape => (1, n_samples)
        # We'll keep it as a CPU tensor for slicing
        waveform = waveform.cpu()
        n_samples = waveform.shape[-1]

        # ----------------------------------------------------------------------
        # 2) Define sliding-window parameters in *samples*
        # ----------------------------------------------------------------------
        window_size = int((window_ms / 1000.0) * sr)  # e.g., 500ms => 8000
        step_size = int((step_ms / 1000.0) * sr)  # e.g., 250ms => 4000

        # If the audio is shorter than one window, we will still handle it:
        # We'll do just one chunk in that case.

        # ----------------------------------------------------------------------
        # 3) Slide over the entire waveform
        # ----------------------------------------------------------------------
        predicted_windows = []
        start = 0
        while True:
            end = start + window_size
            if start >= n_samples:
                break

            # Slice out the chunk => shape (1, chunk_len)
            chunk = waveform[..., start:end]  # (1, chunk_len)

            # Perform inference on this chunk (0.5s)
            pred_idx, pred_label = self.infer_window(chunk)

            # Store (start_time_seconds, end_time_seconds, pred_idx, pred_label)
            start_sec = start / sr
            end_sec = min(end, n_samples) / sr

            predicted_windows.append((start_sec, end_sec, pred_idx, pred_label))

            # Move the sliding window forward
            start += step_size
            if start >= n_samples:
                break

        # ----------------------------------------------------------------------
        # 4) Plot the entire waveform and overlay windows that predicted index=7
        #    (which is the label "off" in your index_to_label list).
        # ----------------------------------------------------------------------
        # Convert waveform to numpy for plotting
        waveform_np = waveform.squeeze(0).numpy()  # shape = (n_samples,)
        time_axis = np.arange(n_samples) / float(sr)

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, waveform_np, label="Audio waveform", color="blue")

        # Overlay windows with predicted_idx == 7
        # We'll color them red with some transparency (alpha)
        for (st, ed, pidx, plabel) in predicted_windows:
            if pidx == 7:  # "off"
                plt.axvspan(st, ed, color="red", alpha=0.3)

        plt.title(f"Sliding-Window Prediction (Highlight = label index 7)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Example usage
    # --------------------------------------------------------------------------
    model_path = "./model.pth"  # Path to your trained BCResNet model
    wav_path = "E:\\CodeRepos\\QuietCuff\\output\\audio_output_20240612_102952_873785_11.wav"

    # Create the inference object
    inference_engine = AudioInference(model_path=model_path, tau=8)

    # Run inference with a 500ms window, 250ms step
    inference_engine(
        wav_path=wav_path,
        window_ms=500,
        step_ms=250
    )
