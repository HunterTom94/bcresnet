#!/usr/bin/env python3

"""
trainer.py

A script that trains a BC-ResNet model on the Google Speech Commands dataset (v1 or v2),
logs training/validation metrics via TensorBoard, and additionally tracks performance on
a specified class (e.g., class 7) over epochs.

Copyright (c) 2023 Qualcomm Technologies, Inc.
All Rights Reserved.
"""

import os
import shutil
import numpy as np
from glob import glob
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils import (
    DownloadDataset,
    SplitDataset,
    SpeechCommand,
    Padding,
    Preprocess,
)

class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters,
        data loaders, and TensorBoard writer.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="Google Speech Command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--tau", default=8, help="Model size scaling factor", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="GPU device ID", type=int)
        parser.add_argument("--download", help="Download data", action="store_true")
        args = parser.parse_args()

        self.save_path = "./model.pth"
        self.__dict__.update(vars(args))

        # Device setup
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.ver = 2

        # TensorBoard writer
        self.writer = SummaryWriter()

        # Data & model
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Main training loop.

        Trains the model for a fixed number of epochs, logs training and validation metrics,
        and finally tests on the official test set.
        """
        # Hyperparameters
        total_epoch = 200
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # Optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9
        )

        # Warmup and total iteration counts
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # Training loop
        for epoch in range(total_epoch):
            self.model.train()
            running_loss = 0.0
            num_samples = 0

            for sample in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epoch}"):
                iterations += 1
                # Learning rate schedule (cosine decay with warmup)
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1 + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward & backward pass
                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Data augmentation / preprocessing
                inputs = self.preprocess_train(inputs, labels, augment=True)

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                # Accumulate loss for this epoch
                running_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)

            # Compute average loss for this epoch
            epoch_loss = running_loss / num_samples
            self.writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            self.writer.add_scalar("Train/LearningRate", lr, epoch)
            print(f"[Epoch {epoch+1:3d}] LR: {lr:.4f}, Train Loss: {epoch_loss:.4f}")

            # Validation
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                self.writer.add_scalar("Validation/Accuracy", valid_acc, epoch)
                print(f"Validation Accuracy: {valid_acc:.3f}")

                # Class-specific performance (e.g., class 7)
                class_7_acc = self.test_class_performance(
                    loader=self.valid_loader,
                    class_of_interest=7,
                    augment=True
                )
                self.writer.add_scalar("Validation/Accuracy_class7", class_7_acc, epoch)
                print(f"Validation Accuracy (Class 7): {class_7_acc:.3f}")

        # Test on the official test set
        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)
        self.writer.add_scalar("Test/Accuracy", test_acc, total_epoch)
        print(f"Test Accuracy: {test_acc:.3f}")

        # Save the final model
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to: {self.save_path}")

        # Close the TensorBoard writer
        self.writer.close()
        print("Training complete.")

    def Test(self, dataset, loader, augment=False):
        """
        Evaluate the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching.
            augment (bool): Whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()

        acc = (true_count / num_testdata) * 100.0
        return acc

    def test_class_performance(self, loader, class_of_interest=7, augment=False):
        """
        Compute and return the accuracy for `class_of_interest` on the given loader.

        Parameters:
            loader (DataLoader): Data loader for the evaluation set.
            class_of_interest (int): The specific class index to track.
            augment (bool): Whether to apply data augmentation (may not be typical in validation).

        Returns:
            float: The accuracy (%age) on the samples belonging to class_of_interest.
        """
        class_correct = 0
        class_total = 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Filter out samples not from class_of_interest
            class_mask = (labels == class_of_interest)
            if class_mask.sum() == 0:
                # No samples of the target class in this batch
                continue

            inputs_class = inputs[class_mask]
            labels_class = labels[class_mask]

            # Preprocess only the relevant samples
            inputs_class = self.preprocess_test(
                inputs_class,
                labels=labels_class,
                is_train=False,
                augment=augment
            )
            outputs_class = self.model(inputs_class)
            preds = torch.argmax(outputs_class, dim=-1)

            class_correct += (preds == labels_class).sum().item()
            class_total += labels_class.size(0)

        if class_total == 0:
            return 0.0

        return float(class_correct) / float(class_total) * 100.0

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Checking Google Speech Commands dataset v1 or v2 ...")

        if not os.path.isdir("./data"):
            os.mkdir("./data")

        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"

        # Override if using version 2
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")

        test_dir = base_dir.replace("commands", "commands_test_set")

        # Download logic
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            # Optionally split the dataset
            SplitDataset(base_dir)
            print("Done downloading/splitting data. Exiting.")
            exit(0)

        # Prepare train/valid/test directories
        train_dir = f"{base_dir}/train_12class"
        valid_dir = f"{base_dir}/valid_12class"
        noise_dir = f"{base_dir}/_background_noise_"

        # Transforms
        transform = transforms.Compose([Padding()])

        # Datasets
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        self.test_dataset  = SpeechCommand(test_dir, self.ver, transform=transform)

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=100, num_workers=0)

        print(
            "Number of samples (train/valid/test): "
            f"{len(self.train_dataset)}/{len(self.valid_dataset)}/{len(self.test_dataset)}"
        )

        # Preprocessing / data augmentation setup
        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print(f"Model: BC-ResNet-{self.tau:.1f} on data v0.0{self.ver}")
        self.model = BCResNets(int(self.tau * 8)).to(self.device)


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
