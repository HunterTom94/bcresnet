#!/usr/bin/env python3

import os
import glob
import random

import librosa
import soundfile as sf
import numpy as np

def load_true_intervals(label_file):
    """
    Read the label file and return a list of (start_sec, end_sec, region_index).
    The region_index is 1-based (i.e., first line is region_index=1).
    """
    intervals = []
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = parts
                if label.lower() == 'b':  # TRUE region
                    start_sec = float(start)
                    end_sec = float(end)
                    intervals.append((start_sec, end_sec, i + 1))
    return intervals

def overlaps_any_true_region(start_s, end_s, true_intervals):
    """
    Given a window [start_s, end_s] and a list of true intervals,
    return the list of region indices that overlap (can be empty).
    Overlap is counted if the window overlaps with at least 50% of a TRUE region's duration.
    """
    overlapping_region_indices = []
    for (region_start, region_end, region_idx) in true_intervals:
        region_duration = region_end - region_start

        # Calculate the overlap duration
        overlap_start = max(start_s, region_start)
        overlap_end = min(end_s, region_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        # Check if the overlap is at least 50% of the region's duration
        if overlap_duration >= 0.3 * region_duration:
            overlapping_region_indices.append(region_idx)

    return overlapping_region_indices

def main():
    # -------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------
    input_folder = './track_label'
    output_folder = './prepared_all'
    os.makedirs(output_folder, exist_ok=True)

    # delete all files in the output_folder
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))

    # The subfolders for each split:
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    for folder in [train_folder, val_folder, test_folder]:
        heartbeat_folder = os.path.join(folder, 'heartbeat')
        non_heartbeat_folder = os.path.join(folder, 'non_heartbeat')
        os.makedirs(heartbeat_folder, exist_ok=True)
        os.makedirs(non_heartbeat_folder, exist_ok=True)

    # Window parameters
    window_size_s = 0.200  # 200 ms
    step_size_s = 0.050  # 50 ms
    target_sr = 16000

    # Train/Val/Test split ratios
    train_ratio = 0.80
    val_ratio = 0.10
    test_ratio = 0.10

    # -------------------------------------------------------------------
    # Gather all wav files
    # -------------------------------------------------------------------
    wav_files = glob.glob(os.path.join(input_folder, '*.wav'))
    if not wav_files:
        print(f"No WAV files found in {input_folder}. Exiting.")
        return

    # Shuffle them so random subset goes to train/val/test
    random.seed(1234)  # for reproducibility; change or remove for a new random shuffle
    random.shuffle(wav_files)

    # Determine how many go to each split
    total_files = len(wav_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    # Partition the files
    train_files = wav_files[:train_count]
    val_files = wav_files[train_count:train_count + val_count]
    test_files = wav_files[train_count + val_count:]

    print(f"Total .wav files: {total_files}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # -------------------------------------------------------------------
    # Helper function to process a single .wav file
    # -------------------------------------------------------------------
    def process_file(wav_path, out_folder):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Corresponding label file
        label_path = os.path.join(input_folder, base_name + '.txt')
        if not os.path.isfile(label_path):
            print(f"WARNING: No label file found for {wav_path}; skipping.")
            return

        # Load the list of TRUE intervals (in seconds)
        true_intervals = load_true_intervals(label_path)

        # Load audio (in original sr), then resample if needed
        y, sr = librosa.load(wav_path, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        duration_s = len(y) / sr

        window_count = 0
        start_s = 0.0

        while True:
            end_s = start_s + window_size_s
            if end_s > duration_s:
                break  # no more windows can fit

            # Overlap check
            region_indices = overlaps_any_true_region(start_s, end_s, true_intervals)

            # Slice the audio
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            snippet = y[start_sample:end_sample]

            # Construct output filename
            window_count += 1
            if len(region_indices) == 0:
                # FALSE label
                out_folder_path = os.path.join(out_folder, 'non_heartbeat')
                out_filename = f"{base_name}_win{window_count:04d}_FALSE.wav"
            else:
                # TRUE label => include all region indices
                out_folder_path = os.path.join(out_folder, 'heartbeat')
                indices_str = "_".join(str(idx) for idx in region_indices)
                out_filename = f"{base_name}_win{window_count:04d}_regions_{indices_str}.wav"

            out_path = os.path.join(out_folder_path, out_filename)

            # Write snippet
            sf.write(out_path, snippet, sr, subtype='PCM_16')

            # Advance window
            start_s += step_size_s

    # -------------------------------------------------------------------
    # Process each split
    # -------------------------------------------------------------------
    for f in train_files:
        process_file(f, train_folder)

    for f in val_files:
        process_file(f, val_folder)

    for f in test_files:
        process_file(f, test_folder)

    print("Processing and splitting complete.")

if __name__ == "__main__":
    main()
