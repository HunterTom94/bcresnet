#!/usr/bin/env python3

import os
import glob
import librosa
import soundfile as sf
import numpy as np


def load_true_intervals(label_file):
    """
    Read the label file and return a list of (start_sec, end_sec)
    for lines labeled 'b'. (1-based indexing for region indices.)
    """
    intervals = []
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = parts
                if label == 'b':  # TRUE region
                    start_sec = float(start)
                    end_sec = float(end)
                    # Store (start, end, region_index)
                    intervals.append((start_sec, end_sec, i + 1))  # i+1 => 1-based
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
        elif overlap_start >= region_start and overlap_end <= region_end:
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

    # Window/step in seconds
    window_size_s = 0.200
    step_size_s = 0.050
    target_sr = 16000

    # -------------------------------------------------------------------
    # Process each .wav / .txt pair
    # -------------------------------------------------------------------
    # Find all .wav files in the input folder
    wav_files = glob.glob(os.path.join(input_folder, '*.wav'))

    for wav_path in wav_files:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Corresponding label file
        label_path = os.path.join(input_folder, base_name + '.txt')
        if not os.path.isfile(label_path):
            print(f"WARNING: No label file found for {wav_path}, skipping.")
            continue

        # Load the list of TRUE intervals (in seconds)
        true_intervals = load_true_intervals(label_path)

        # Read audio and resample to 16 kHz
        y, sr = librosa.load(wav_path, sr=None)  # load in original sr
        if sr != target_sr:
            # Resample to target_sr
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        else:
            # Already at target_sr
            pass

        duration_s = len(y) / sr

        # Sliding window extraction
        window_count = 0
        start_s = 0.0

        while True:
            end_s = start_s + window_size_s
            if end_s > duration_s:
                break  # no more windows can fit

            # Determine if this window overlaps any TRUE region
            region_indices = overlaps_any_true_region(start_s, end_s, true_intervals)

            # Slice the audio
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            snippet = y[start_sample:end_sample]

            # Construct output filename
            window_count += 1
            if len(region_indices) == 0:
                # FALSE label
                out_filename = f"{base_name}_win{window_count:04d}_FALSE.wav"
            else:
                # TRUE label => include all region indices
                # e.g. if region_indices = [1,3], becomes "1_3"
                indices_str = "_".join(str(idx) for idx in region_indices)
                out_filename = f"{base_name}_win{window_count:04d}_regions_{indices_str}.wav"

            out_path = os.path.join(output_folder, out_filename)

            # Write the snippet to disk
            sf.write(out_path, snippet, sr, subtype='PCM_16')

            # Move to next window
            start_s += step_size_s

    print("Processing complete.")


if __name__ == "__main__":
    main()
