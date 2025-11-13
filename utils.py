import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os
import numpy as np
from sklearn.metrics import roc_curve
from pathlib import Path
import librosa

# This print statement MUST appear when you run the script.
print("--- [DEBUG] v7 utils.py with 'original_gen' typo fixed. ---")

# --- 1. Dataset Class ---

class AudioLogSpecDataset(Dataset):
    """
    Loads (raw_audio, log_spec_tensor, label) from a JSON file.
    Uses ONE 'dataset_root_dir' which contains:
    - training/real/... (audio)
    - training/fake/... (audio)
    - log_specs/training/real/... (spectrograms)
    """
    def __init__(self, json_file_path, dataset_root_dir, max_audio_len=80000): # 5 sec @ 16kHz
        # This is the root for ALL files, e.g., D:\...fake_real
        self.dataset_root = Path(dataset_root_dir)
        
        # json_file_path is the *absolute* path, e.g., D:\...fake_real\train.json
        json_path = Path(json_file_path)
        
        print(f"  [Dataset] Using all-in-one root: {self.dataset_root}")
        
        try:
            with open(json_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: Cannot find JSON file at {json_path}")
            print("Please check the --train_json and --val_json paths.")
            raise
            
        self.max_audio_len = max_audio_len
        self.hop_length = 160 # Must match preprocessing
        self.spec_max_frames = max_audio_len // self.hop_length 
        
        print(f"Loaded {len(self.metadata)} samples from {json_path}")
        print(f"Max audio len: {self.max_audio_len} samples, Max spec frames: {self.spec_max_frames}")


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path_rel, spec_path_rel, label = self.metadata[idx]
        
        # Path 1: Audio
        # Joins D:\...fake_real \ training\real\file.wav
        audio_path = self.dataset_root / audio_path_rel
        
        # Path 2: Spectrogram
        # Joins D:\...fake_real \ log_specs\training\real\file.pt
        spec_path = self.dataset_root / spec_path_rel
        
        try:
            # --- Load Full Audio (with librosa fallback) ---
            waveform, sample_rate = None, None
            librosa_fallback_used = False
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                error_str = str(e).lower()
                if "flac" in error_str or "backend" in error_str or "unknown" in error_str or "mp3" in error_str or "system error" in error_str:
                    try:
                        waveform_np, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
                        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                        librosa_fallback_used = True
                    except Exception as librosa_e:
                        print(f"DEBUG: Error loading {audio_path_rel}. Librosa fallback failed: {librosa_e}. Returning dummy.")
                        raise
                else:
                    print(f"DEBUG: Error loading {audio_path_rel}. Unexpected torchaudio error: {e}. Returning dummy.")
                    raise
            
            # --- Load Spectrogram ---
            log_spec = torch.load(spec_path) # (n_mels, n_frames)
            
            # --- Pre-processing (Resample, Mono) ---
            if not librosa_fallback_used:
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                else:
                    waveform = waveform.squeeze(0) # (num_samples)
            else:
                waveform = waveform.squeeze(0)

            original_len = waveform.shape[0]
            spec_original_len = log_spec.shape[1]

            # --- SYNCHRONIZED Padding / Cropping ---
            if original_len > self.max_audio_len:
                # --- V7 FIX: Corrected 'original_gen' to 'original_len' ---
                start_sample = np.random.randint(0, original_len - self.max_audio_len + 1)
                end_sample = start_sample + self.max_audio_len
                waveform_padded = waveform[start_sample : end_sample]
                attention_mask = torch.ones(self.max_audio_len, dtype=torch.long)
            else:
                start_sample = 0
                waveform_padded = F.pad(waveform, (0, self.max_audio_len - original_len), 'constant', 0)
                attention_mask = torch.zeros(self.max_audio_len, dtype=torch.long)
                attention_mask[:original_len] = 1

            start_frame = start_sample // self.hop_length
            end_frame = start_frame + self.spec_max_frames
            
            if end_frame > spec_original_len:
                log_spec_padded = F.pad(log_spec, (0, end_frame - spec_original_len), 'constant', 0)
            else:
                log_spec_padded = log_spec
                
            log_spec_cropped = log_spec_padded[:, start_frame : end_frame]
            
            if log_spec_cropped.shape[1] != self.spec_max_frames:
                 log_spec_cropped = F.pad(log_spec_cropped, (0, self.spec_max_frames - log_spec_cropped.shape[1]), 'constant', 0)

            return waveform_padded, attention_mask, log_spec_cropped, int(label)
        
        except Exception as e:
            # Check for file not found specifically
            if isinstance(e, FileNotFoundError):
                 print(f"FATAL: File not found. Check paths.")
                 print(f"  Audio path: {audio_path}")
                 print(f"  Spec path: {spec_path}")
                 print("  Is your --data_root correct?")
            
            print(f"Error loading index {idx} ({audio_path_rel} or {spec_path_rel}): {e}. Returning dummy data.")
            return (
                torch.zeros(self.max_audio_len), 
                torch.zeros(self.max_audio_len, dtype=torch.long), 
                torch.zeros(128, self.spec_max_frames), 
                0
            )

# --- 2. Collate Function ---
def collate_fn_pad(batch):
    raw_waveforms = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    spec_tensors = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    raw_waveforms_padded = torch.stack(raw_waveforms)
    attention_masks_padded = torch.stack(attention_masks)
    spec_tensors_padded = torch.stack(spec_tensors)
    labels_tensor = torch.LongTensor(labels)
    
    return raw_waveforms_padded, attention_masks_padded, spec_tensors_padded, labels_tensor

# --- 3. OC-Softmax Loss Function ---
class OCSoftmaxLoss(nn.Module):
    def __init__(self, in_features, m0=0.9, m1=0.2, alpha=20.0):
        super(OCSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.m0 = m0
        self.m1 = m1
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(in_features))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, embeddings, labels):
        # --- MY BUG FIX: Changed embeddings__norm (two underscores) to embeddings_norm (one) ---
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=0)
        
        cosine_sim = embeddings_norm.matmul(weight_norm)
        
        margins = torch.full_like(cosine_sim, self.m1)
        margins[labels == 0] = self.m0
        
        y_pow = torch.pow(-1.0, labels.float())
        
        loss = torch.log(1 + torch.exp(self.alpha * (margins - cosine_sim) * y_pow))
        
        return loss.mean(), cosine_sim.detach()

# --- 4. EER Calculation ---
def compute_eer(bonafide_scores, spoof_scores):
    """Calculates the Equal Error Rate (EER) with robustness."""
    if not bonafide_scores or not spoof_scores:
        # print("Warning: EER calculation failed (no bonafide or spoof scores).")
        return 100.0
        
    labels = [0] * len(bonafide_scores) + [1] * len(spoof_scores)
    scores = bonafide_scores + spoof_scores
    
    try:
        # Note: pos_label=0 means 'bona fide' is the positive class
        # This is standard for ASV/CM EER calculation
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0) 
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
        return eer * 100
    except Exception as e:
        # print(f"Warning: EER calculation failed: {e}. Returning 100.0")
        return 100.0 # Return 100% EER on failure

