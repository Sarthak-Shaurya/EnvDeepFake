import torch
from tqdm import tqdm
from utils import compute_eer
import numpy as np
import logging

def train_one_epoch(model, dataloader, criterion, optimizer, device, logger=None):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, (raw_padded, raw_mask, spec_padded, labels) in enumerate(pbar):
        raw_padded = raw_padded.to(device)
        raw_mask = raw_mask.to(device)
        spec_padded = spec_padded.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        embeddings = model(raw_padded, raw_mask, spec_padded)
        loss, _ = criterion(embeddings, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        current_avg_loss = running_loss / (i + 1)
        pbar.set_postfix(loss=f"{current_avg_loss:.4f}")

        if logger and (i + 1) % 100 == 0:
            logger.info(f"Batch {i + 1}/{len(dataloader)}: Current Avg. Loss: {current_avg_loss:.4f}")
        
    final_avg_loss = running_loss / len(dataloader)
    if logger:
        logger.info(f"Training epoch finished. Average Loss: {final_avg_loss:.4f}")
    return final_avg_loss

def validate(model, dataloader, criterion, device, logger=None):
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for (raw_padded, raw_mask, spec_padded, labels) in pbar:
            raw_padded = raw_padded.to(device)
            raw_mask = raw_mask.to(device)
            spec_padded = spec_padded.to(device)
            labels = labels.to(device)
            
            embeddings = model(raw_padded, raw_mask, spec_padded)
            
            loss, scores = criterion(embeddings, labels)
            running_loss += loss.item()
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    bonafide_scores = all_scores[all_labels == 0].tolist()
    spoof_scores = all_scores[all_labels == 1].tolist()
    
    val_eer = compute_eer(bonafide_scores, spoof_scores)
    
    if logger:
        logger.info(f"Validation complete. Loss: {val_loss:.4f}, EER: {val_eer:.2f}%")

    return val_loss, val_eer
