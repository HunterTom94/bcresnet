# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

# -------------------------
# Added import for TensorBoard
# -------------------------
from torch.utils.tensorboard import SummaryWriter

# Import your BCResNet model (unchanged)
from bcresnet import BCResNets

# Import your custom dataset utilities (modified below)
from utils_2_classes import TwoClassDataset, Padding, Preprocess

class Trainer:
    def __init__(self):
        """
        Trainer for a 2-class heartbeat vs. non_heartbeat classification
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--data_dir",
            default="G:\\CodeRepo\\k-sound-data\\prepared_all",
            help="Path to the root data directory (with train/valid/test)",
            type=str,
        )
        parser.add_argument(
            "--batch_size", default=64, help="Batch size", type=int
        )
        parser.add_argument(
            "--tau", default=8, help="Model size factor for BCResNet", type=float
        )
        parser.add_argument("--gpu", default=0, help="GPU device ID", type=int)
        args = parser.parse_args()

        self.save_path = "./model_2_classes.pth"
        self.__dict__.update(vars(args))

        # Choose device
        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        )

        # -------------------------
        # Initialize TensorBoard writer
        # -------------------------
        self.writer = SummaryWriter()

        # Load data and model
        self._load_data()
        self._load_model()

    def _load_data(self):
        """
        Sets up train/valid/test data loaders.
        Expects directory structure:
            data/
               ├── train/
               │    ├── heartbeat/
               │    └── non_heartbeat/
               ├── valid/
               └── test/
        """
        print("Loading 2-class dataset (heartbeat vs. non_heartbeat) ...")

        # Some basic transform: zero-pad to 1 second
        transform = Padding()

        train_dataset = TwoClassDataset(
            os.path.join(self.data_dir, "train"),
            transform=transform
        )
        print(f"Train folder path: {os.path.join(self.data_dir, 'train')}")

        valid_dataset = TwoClassDataset(
            os.path.join(self.data_dir, "val"),
            transform=transform
        )
        print(f"Validation dataset length: {len(valid_dataset)}")  # Debug print

        test_dataset = TwoClassDataset(
            os.path.join(self.data_dir, "test"),
            transform=transform
        )
        print(f"Test dataset length: {len(test_dataset)}")  # Debug print

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        print("Data sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Valid: {len(valid_dataset)}")
        print(f"  Test:  {len(test_dataset)}")

        # Example: enable spec-augment if tau >= 1.5
        specaugment = self.tau >= 1.5
        frequency_masking_para = {
            1.0: 0,
            1.5: 5,
            2.0: 7,
            3.0: 7,
            6.0: 7,
            8.0: 7
        }
        freq_mask_val = frequency_masking_para.get(self.tau, 0)

        # Create Preprocess objects for train/test
        self.preprocess_train = Preprocess(
            noise_loc=None,  # or point to your noise folder
            device=self.device,
            specaug=specaugment,
            frequency_masking_para=freq_mask_val,
        )
        self.preprocess_test = Preprocess(
            noise_loc=None,
            device=self.device,
            specaug=False,  # no spec-augment for valid/test
        )

    def _load_model(self):
        """
        Create BCResNet model based on tau
        """
        print(f"Loading BCResNet with tau={self.tau}")
        channels = int(self.tau * 8)  # for BCResNet
        self.model = BCResNets(channels, num_classes=2)  # 2-class
        self.model = self.model.to(self.device)

    def __call__(self):
        """
        Main training loop
        """
        total_epoch = 200
        warmup_epoch = 2
        init_lr = 0.01
        lr_lower_limit = 0

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0,
            weight_decay=1e-3,
            momentum=0.9,
        )

        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        for epoch in range(total_epoch):
            self.model.train()
            running_loss = 0.0
            num_samples = 0

            for sample in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epoch}"):
                iterations += 1

                # Cosine LR schedule with warmup
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1 + np.cos(
                            np.pi * (iterations - n_step_warmup)
                            / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Data augmentation & feature extraction
                inputs = self.preprocess_train(
                    inputs, labels, augment=True, is_train=True
                )

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)

            # Compute average epoch loss
            epoch_loss = running_loss / num_samples

            # -------------
            # Log to TensorBoard
            # -------------
            self.writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            self.writer.add_scalar("Train/LearningRate", lr, epoch)

            # Validation step
            valid_acc = self.evaluate(self.valid_loader, augment=False)
            self.writer.add_scalar("Validation/Accuracy", valid_acc, epoch)
            print(f"[Epoch {epoch+1:2d}] LR: {lr:.5f} | "
                  f"Train Loss: {epoch_loss:.4f} | Valid Acc: {valid_acc:.2f}")

        # Final test set
        test_acc = self.evaluate(self.test_loader, augment=False)
        self.writer.add_scalar("Test/Accuracy", test_acc, total_epoch)
        print(f"Final Test Acc = {test_acc:.2f}")

        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to: {self.save_path}")

        # -------------
        # Close the TensorBoard writer
        # -------------
        self.writer.close()
        print("End of Training.")

    def evaluate(self, loader, augment=False):
        """
        Evaluate model on a loader
        """
        self.model.eval()
        true_count = 0.0
        total_count = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # For valid/test, we typically do not do heavy augmentation
                inputs = self.preprocess_test(
                    inputs, labels=labels, augment=augment, is_train=False
                )

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=-1)
                true_count += (preds == labels).sum().item()
                total_count += labels.size(0)

        return 100.0 * true_count / total_count


if __name__ == "__main__":
    trainer = Trainer()
    trainer()