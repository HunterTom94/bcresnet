import torch
import torchaudio
import torchaudio.transforms as T
import os

from bcresnet import BCResNets
from utils import Padding, Preprocess


# Make sure to have these local imports or place them accordingly:
# from bcresnet import BCResNets
# from utils import Padding, Preprocess

# If you do not have the exact same environment,
# ensure the "bcresnet.py" and "utils.py" are accessible.

class AudioInference:
    def __init__(self, model_path, tau=8, device=None, noise_dir="./data/speech_commands_v0.02/_background_noise_"):
        """
        Initialize the inference class.

        :param model_path: Path to the saved .pth file
        :param tau: Must match the tau used during training (default=8)
        :param device: 'cuda' or 'cpu'; if None, will auto-detect
        :param noise_dir: Path to background noise folder,
                          should match the directory used in training.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = BCResNets(int(tau * 8)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # For Google Speech Commands, the typical sample rate is 16k
        self.sample_rate = 16000

        # We use the same transform as in training for zero-padding
        self.padding_transform = Padding()

        # We set up the same test-time Preprocess
        # (no frequency masking at test time, so set frequency_masking_para=0)
        self.preprocess = Preprocess(noise_dir, self.device, specaug=False, frequency_masking_para=0)

        # You might want a mapping from class indices to labels:
        # These are the 12 standard classes from the dataset
        # (ordered as in 'SpeechCommand' dataset in `utils.py`).
        # If you changed the classes or ordering in your custom dataset,
        # be sure to update accordingly.
        self.index_to_label = [
            "yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go", "unknown", "silence"
        ]

    def __call__(self, wav_path):
        """
        Perform inference on a single audio file (any length).

        :param wav_path: Path to the audio file
        :return: Predicted label
        """
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Load the audio
        waveform, sr = torchaudio.load(wav_path)

        # Resample if needed to 16k
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert stereo to mono if necessary (take the first channel)
        # If waveform is [channels, time], keep [0, :]
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        # The training code expects shape: (batch, 1, time).
        # So let's unsqueeze batch dimension => shape: (1, 1, time)
        waveform = waveform.unsqueeze(0)

        # We must zero-pad (or clip) to 1s if it is any length
        # For short audio: zero-pad to 16000
        # For long audio: we will simply *clip* to first 16000 samples
        # to keep consistent with training.
        if waveform.shape[-1] > self.sample_rate:
            waveform = waveform[..., : self.sample_rate]

        # Apply Padding transform to ensure size=16000
        # (This transform zero-pads if under 16000 samples)
        waveform = self.padding_transform(waveform)

        # Now move to device
        waveform = waveform.to(self.device)

        # We do not need label information at test time
        # but Preprocess function signature is (inputs, labels, is_train, augment).
        # We pass labels=None or a dummy value. is_train=False, augment=False.
        waveform = self.preprocess(waveform, labels=None, is_train=False, augment=False)

        # Forward pass
        with torch.no_grad():
            logits = self.model(waveform)
            prediction = torch.argmax(logits, dim=-1).item()

        # Map class index to text label
        predicted_label = self.index_to_label[prediction]

        return predicted_label


if __name__ == "__main__":
    # Example usage
    model_path = "./model.pth"  # Path to your trained model file
    wav_path = "E:\\CodeRepos\\bcresnet\\bcresnet\\heart_beat_data\\test\\segment_normal__128_1306344005749_D_0.wav"  # Path to your WAV/PCM/other audio

    # Create inference object
    inference_engine = AudioInference(model_path=model_path, tau=8)

    # Run inference
    label = inference_engine(wav_path)
    print(f"Predicted label for '{wav_path}': {label}")
