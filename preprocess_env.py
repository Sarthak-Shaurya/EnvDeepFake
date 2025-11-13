import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing
import sys
import warnings
import random
import numpy as np

# --- NEW IMPORT ---
# Librosa will be our fallback audio loader
import librosa 

# Set random seed for reproducibility in splitting
random.seed(42)

# --- Global/Shared Parameters (MODIFIED to match ASVspoof) ---
SR = 16000
N_FFT = 1024
HOP_LENGTH = 160 # 10ms hop @ 16kHz
N_MELS = 128
# F_MAX removed to match asvspoof

# --- Mel Spectrogram Transformer (from ASVspoof) ---
mel_transformer = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

# --- Custom Augmentation Functions (KEPT UNCHANGED) ---
def apply_add_gaussian_noise(waveform, sample_rate):
    """Adds random Gaussian noise to the waveform."""
    min_amplitude = 0.001
    max_amplitude = 0.015
    amplitude = random.uniform(min_amplitude, max_amplitude)
    noise = torch.randn_like(waveform)
    return waveform + noise * amplitude

def apply_time_stretch(waveform, sample_rate):
    """Stretches the time of the audio without changing the pitch."""
    min_rate = 0.8
    max_rate = 1.25
    rate = random.uniform(min_rate, max_rate)
    # This requires spectrogram -> stretch -> inverse spectrogram
    spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)(waveform)
    stretcher = torchaudio.transforms.TimeStretch(n_freq=N_FFT // 2 + 1, hop_length=HOP_LENGTH, fixed_rate=rate)
    stretched_spec = stretcher(spec.type(torch.complex64))
    inverter = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)
    return inverter(stretched_spec)

def apply_pitch_shift(waveform, sample_rate):
    """Shifts the pitch of the audio."""
    min_semitones = -4
    max_semitones = 4
    semitones = random.uniform(min_semitones, max_semitones)
    return torchaudio.functional.pitch_shift(waveform, sample_rate=sample_rate, n_steps=semitones)

def apply_random_gain(waveform, sample_rate):
    """Applies a random gain in decibels."""
    min_gain_db = -6
    max_gain_db = 6
    gain_db = random.uniform(min_gain_db, max_gain_db)
    return torchaudio.functional.gain(waveform, gain_db)

def apply_polarity_inversion(waveform, sample_rate):
    """Flips the amplitude of the waveform."""
    return waveform * -1.0

def apply_high_pass_filter(waveform, sample_rate):
    """Applies a high-pass filter."""
    min_freq = 200
    max_freq = 2000
    cutoff_freq = random.uniform(min_freq, max_freq)
    return torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq)

def apply_low_pass_filter(waveform, sample_rate):
    """Applies a low-pass filter."""
    min_freq = 500
    max_freq = 4000
    cutoff_freq = random.uniform(min_freq, max_freq)
    return torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq)

def apply_shift(waveform, sample_rate):
    """Shifts the audio waveform circularly."""
    min_fraction = -0.5
    max_fraction = 0.5
    shift_fraction = random.uniform(min_fraction, max_fraction)
    num_samples = waveform.shape[-1]
    shift_samples = int(num_samples * shift_fraction)
    return torch.roll(waveform, shifts=shift_samples, dims=-1)
    
def apply_noise_and_pitch(waveform, sample_rate):
    """Applies Gaussian noise and then pitch shift."""
    augmented_wav = apply_add_gaussian_noise(waveform, sample_rate)
    return apply_pitch_shift(augmented_wav, sample_rate)

def apply_pitch_and_hpf(waveform, sample_rate):
    """Applies pitch shift and then a high-pass filter."""
    augmented_wav = apply_pitch_shift(waveform, sample_rate)
    return apply_high_pass_filter(augmented_wav, sample_rate)

def apply_noise_and_hpf(waveform, sample_rate):
    """Applies Gaussian noise and then a high-pass filter."""
    augmented_wav = apply_add_gaussian_noise(waveform, sample_rate)
    return apply_high_pass_filter(augmented_wav, sample_rate)

AUGMENTATIONS = {
    "white_noise": apply_add_gaussian_noise, "time_stretch": apply_time_stretch,
    "pitch_scale": apply_pitch_shift, "random_gain": apply_random_gain,
    "invert_polarity": apply_polarity_inversion, "high_pass": apply_high_pass_filter,
    "low_pass": apply_low_pass_filter, "shift": apply_shift,
    "noise_and_pitch": apply_noise_and_pitch, "pitch_and_hpf": apply_pitch_and_hpf,
    "noise_and_hpf": apply_noise_and_hpf,
}

# --------------------------------------------------------------------------------
# --- Worker function (MODIFIED to match ASVspoof) ---
# --------------------------------------------------------------------------------

def process_single_file(args):
    """
    Worker function to process a single audio file, save its Mel spectrogram tensor.
    Matches the logic of preprocess_asvspoof.py.
    """
    # MODIFIED argument unpacking
    audio_path_str, label, output_dir_root_str, dataset_root_str, aug_key, file_id_for_naming = args
    
    audio_path = Path(audio_path_str)
    dataset_root = Path(dataset_root_str)
    output_dir_root = Path(output_dir_root_str)

    try:
        # 1. Generate paths (Logic from asvspoof)
        audio_rel_path = str(audio_path.relative_to(dataset_root))
        
        # Use the unique file_id for the stem
        spec_filename = f"{file_id_for_naming}.pt"
            
        spec_rel_dir = os.path.dirname(audio_rel_path)
        # Save to output_dir_root / "log_specs" / ...relative_path...
        spec_save_dir = output_dir_root / "log_specs" / spec_rel_dir
        spec_save_dir.mkdir(parents=True, exist_ok=True)
        spec_save_path = spec_save_dir / spec_filename

        # 2. Load Audio (Logic from asvspoof)
        waveform, sample_rate = None, None
        librosa_fallback_used = False
        try:
            # --- Primary method: torchaudio ---
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # --- Fallback method: librosa ---
            error_str = str(e).lower()
            # Added "wav" to the list of errors to catch
            if "flac" in error_str or "backend" in error_str or "unknown" in error_str or "mp3" in error_str or "system error" in error_str or "wav" in error_str:
                try:
                    # Librosa loads as numpy, resamples to target SR, and makes mono
                    waveform_np, sample_rate = librosa.load(str(audio_path), sr=SR, mono=True)
                    # Convert to torch tensor and add channel dim [1, num_samples]
                    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                    librosa_fallback_used = True
                except Exception as librosa_e:
                    # If librosa also fails, we give up on this file
                    error_msg = f"Warning: Librosa ALSO failed for {audio_path}. Skipping. Error: {librosa_e}"
                    print(error_msg, flush=True)
                    return (False, None, error_msg)
            else:
                # This was a different, unexpected error
                error_msg = f"Warning: Unexpected torchaudio error loading {audio_path}. Skipping. Error: {e}"
                print(error_msg, flush=True)
                return (False, None, error_msg) # Skip this file

        if waveform is None:
             # This should not be hit if logic is correct, but as a safeguard
            error_msg = f"Warning: Waveform for {audio_path} is None after load attempts. Skipping."
            print(error_msg, flush=True)
            return (False, None, error_msg)

        # 3. Resample & Mono (Logic from asvspoof)
        if not librosa_fallback_used:
            if sample_rate != SR:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SR)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 4. Apply Augmentation (Logic from _env)
        if aug_key:
            transform_func = AUGMENTATIONS[aug_key]
            waveform = transform_func(waveform=waveform, sample_rate=SR)

        # 5. Create and Save Log-Spectrogram (Logic from asvspoof)
        with torch.no_grad():
            # Run on CPU to avoid OOM issues in workers
            spec = mel_transformer(waveform.to("cpu")) 
            log_spec = amplitude_to_db(spec)
            
            # --- REMOVED [0, 1] normalization to match asvspoof ---
            
            torch.save(log_spec.squeeze(0).cpu(), spec_save_path)
            
        # 6. Return JSON entry [audio_path, spec_path, label] (Logic from asvspoof)
        
        # Path to the saved spectrogram, relative to the output_dir_root
        spec_rel_path = str(spec_save_path.relative_to(output_dir_root))
        # Path to the original audio, relative to the dataset_root
        audio_rel_path_for_json = str(audio_path.relative_to(dataset_root)) 
        
        # Return format: (success, [entry], error_msg)
        return (True, [audio_rel_path_for_json, spec_rel_path, label], None)
        
    except Exception as e:
        # This catches errors in path logic, augmentation, or spec creation
        error_msg = f"Skipping {audio_path} due to: {e}"
        if "memory" in str(e).lower() or "paging file" in str(e).lower() or "alloc_cpu" in str(e).lower() or "winerror 1455" in str(e).lower():
             print(f"\n[!!! CRITICAL MEMORY ERROR !!!] {error_msg}", file=sys.stderr, flush=True)
        else:
             print(f"\n[Worker Error] {error_msg}", file=sys.stderr, flush=True)
        return (False, None, error_msg)

# --------------------------------------------------------------------------------
# --- Other functions (Kept unchanged for structure traversal and splitting) ---
# --------------------------------------------------------------------------------

def find_all_audio_files(root_dir):
    root_path = Path(root_dir)
    real_audio_files = []
    fake_audio_files = []
    
    print(f"Searching for audio files in: {root_dir}")

    # 1. Real Audio (Label 0: Bonafide)
    real_path = root_path / "real_audio"
    if real_path.is_dir():
        for dataset_folder in ["TUTASC2019Dev", "TUTSED2016Dev", "TUTSED2016Eval", 
                               "TUTSED2017Dev", "TUTSED2017Eval", "UrbanSound8K"]:
            folder_path = real_path / dataset_folder
            if folder_path.is_dir():
                for audio_path in folder_path.rglob('*.wav'):
                    real_audio_files.append({"path": str(audio_path), "label": 0, "id": audio_path.stem})
    
    # 2. Fake Audio (Label 1: Spoof/Synthetic)
    fake_path = root_path / "fake_audio"
    if fake_path.is_dir():
        for gen_type in ["TTA", "ATA"]:
            type_path = fake_path / gen_type
            if type_path.is_dir():
                for generator in type_path.iterdir():
                    if generator.is_dir():
                        for dataset_folder in ["TUTASC2019Dev", "TUTSED2016Dev", "TUTSED2016Eval", 
                                               "TUTSED2017Dev", "TUTSED2017Eval", "UrbanSound8K"]:
                            folder_path = generator / dataset_folder
                            if folder_path.is_dir():
                                for audio_path in folder_path.rglob('*.wav'):
                                    # Use a more robust relative ID
                                    relative_id = str(audio_path.relative_to(root_dir)).replace(os.sep, '_').replace('.wav', '')
                                    fake_audio_files.append({"path": str(audio_path), "label": 1, "id": relative_id})

    print(f"Found {len(real_audio_files)} Real audio files.")
    print(f"Found {len(fake_audio_files)} Fake audio files.")
    
    return real_audio_files, fake_audio_files

def stratified_split(real_files, fake_files, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    assert train_ratio + test_ratio + val_ratio == 1.0, "Split ratios must sum to 1.0"
    
    splits = {"train": [], "test": [], "validation": []}
    
    for files in [real_files, fake_files]:
        random.shuffle(files)
        total = len(files)
        
        train_count = int(total * train_ratio)
        test_count = int(total * test_ratio)
        val_count = total - train_count - test_count
        
        splits["train"].extend(files[:train_count])
        splits["test"].extend(files[train_count:train_count + test_count])
        splits["validation"].extend(files[train_count + test_count:])
        
    print("\n--- Split Statistics ---")
    for split_name, file_list in splits.items():
        real_count = sum(1 for f in file_list if f['label'] == 0)
        fake_count = sum(1 for f in file_list if f['label'] == 1)
        print(f"  {split_name.upper():<10}: Total={len(file_list):<6}, Real={real_count:<6}, Fake={fake_count:<6}")
        
    return splits

# --------------------------------------------------------------------------------
# --- Processing function (MODIFIED to match ASVspoof) ---
# --------------------------------------------------------------------------------

def process_split_v2(files_list, output_root, dataset_root, split_name, num_workers):
    print(f"\n--- Processing partition: {split_name.upper()} ---")
    
    # MODIFIED: JSON path to match asvspoof (e.g., train.json)
    output_json_path = output_root / f"{split_name}.json"
    
    # MODIFIED: Removed output_mel_dir, as path is now built in worker
    
    worker_args = []
    
    for file_info in files_list:
        audio_file_path = file_info["path"]
        file_id = file_info["id"]
        label_val = file_info["label"]
        
        # Base entry (no augmentation)
        # MODIFIED: Worker args to match process_single_file
        worker_args.append((audio_file_path, label_val, output_root, dataset_root, None, file_id))
        
        # Augmentation only for 'train' set and for REAL audio (label 0)
        if split_name == 'train' and label_val == 0:
            for aug_key in AUGMENTATIONS.keys():
                aug_file_id = f"{file_id}_aug_{aug_key}"
                # MODIFIED: Worker args to match process_single_file
                worker_args.append((audio_file_path, label_val, output_root, dataset_root, aug_key, aug_file_id))
    
    if not worker_args:
        print(f"No audio files found for partition '{split_name}'. Skipping.")
        return

    # Use the provided number of workers
    print(f"Found {len(worker_args)} tasks (including augmentations). Starting parallel processing with {num_workers} workers...")
    
    processed_entries, skipped_files = [], []

    # Use multiprocessing.Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(worker_args), desc=f"Preprocessing {split_name}") as pbar:
            for result in pool.imap_unordered(process_single_file, worker_args):
                is_success, entry, skipped_msg = result
                if is_success:
                    # entry is already the list [audio_rel, spec_rel, label]
                    processed_entries.append(entry)
                elif skipped_msg:
                    skipped_files.append(skipped_msg)
                pbar.update(1)

    with open(output_json_path, "w") as f:
        # Dump the list of lists directly, as required
        json.dump(processed_entries, f, indent=2)
        
    print(f"✅ Finished {split_name} partition!")
    # MODIFIED: Log message to reflect new save directory
    print(f"   - {len(processed_entries)} mel spectrograms saved in: {output_root / 'log_specs'}")
    print(f"   - Metadata saved to: {output_json_path}")
    if skipped_files:
        print(f"   - ❗️ Skipped {len(skipped_files)} files due to errors.")
        
# --------------------------------------------------------------------------------
# --- Main function (MODIFIED to pass dataset_root) ---
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess custom audio dataset structure into Mel spectrogram tensors and JSON metadata files with a Train/Test/Validation split.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the root directory containing 'real_audio' and 'fake_audio' folders.")
    parser.add_argument("--output_root", type=str, required=True, help="Path to the output directory where processed data will be saved. For compatibility with utils.py, this should be the SAME as --dataset_root.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes to use. Default is 1.") 
    args = parser.parse_args()
    
    # --- IMPORTANT ---
    if args.dataset_root != args.output_root:
        print("="*50)
        print("WARNING:")
        print(f"Your --dataset_root is: {args.dataset_root}")
        print(f"Your --output_root is:   {args.output_root}")
        print("For the provided 'utils.py' to work correctly, these two paths MUST be the same.")
        print("The 'utils.py' script expects 'log_specs' to be inside the 'dataset_root'.")
        print("Proceeding, but training will likely fail if paths are different.")
        print("="*50)
    
    warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")
    warnings.filterwarnings("ignore", message="torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated")
    warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

    dataset_root, output_root = Path(args.dataset_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Traverse and Collect Audio Files
    real_files, fake_files = find_all_audio_files(dataset_root)
    
    if not (real_files or fake_files):
        print("❌ ERROR: No audio files (.wav) found. Check 'dataset_root' path and folder structure.")
        sys.exit(1)
        
    # 2. Perform Stratified Split (e.g., 70/15/15)
    splits = stratified_split(real_files, fake_files, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15)
    
    # 3. Process each split
    for split_name in ["train", "test", "validation"]:
        # MODIFIED: Pass dataset_root to the processor
        process_split_v2(splits[split_name], output_root, dataset_root, split_name, args.workers)
    
    print("\n🎉 All partitions processed successfully!")

if __name__ == "__main__":
    main()