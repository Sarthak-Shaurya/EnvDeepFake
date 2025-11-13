import torch
from tqdm import tqdm
from utils import compute_eer
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve

def test_model(model, dataloader, criterion, device, logger=None):
    """
    Evaluates the model on the test set and prints/logs comprehensive metrics.
    """
    model.eval()
    all_scores = []
    all_labels = []

    # Get the target weight from the loss function for scoring
    # This MUST be the weight from the *trained* criterion
    target_weight = torch.nn.functional.normalize(criterion.weight.data, p=2, dim=0)
    
    eval_desc = "Running evaluation on the test set..."
    if logger:
        logger.info(eval_desc)
    print(eval_desc)

    with torch.no_grad():
        for (raw_padded, raw_mask, spec_padded, labels) in tqdm(dataloader, desc="Testing"):
            raw_padded = raw_padded.to(device)
            raw_mask = raw_mask.to(device)
            spec_padded = spec_padded.to(device)
            
            embeddings = model(raw_padded, raw_mask, spec_padded)
            
            # Compute CM scores (cosine similarity to target weight)
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            scores = embeddings_norm.matmul(target_weight).cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Separate scores for EER calculation
    # High score = bona fide (0), Low score = spoof (1)
    bonafide_scores = all_scores[all_labels == 0].tolist()
    spoof_scores = all_scores[all_labels == 1].tolist()
    
    test_eer = compute_eer(bonafide_scores, spoof_scores)
    
    # --- Calculate Advanced Metrics ---
    
    # Find the threshold at the EER point
    # We use pos_label=0 (bona fide) for EER calculation, as in utils.py
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=0) 
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_index]
    
    # Calculate AUC
    # For AUC, standard practice is to have 1 as the positive class (spoof)
    # Since high scores mean bona fide (0), we need to invert scores for AUC
    # so that high scores mean spoof (1).
    scores_for_auc = -all_scores 
    test_auc = roc_auc_score(all_labels, scores_for_auc)
    
    # Get binary predictions based on the EER threshold
    # Prediction is 1 (spoof) if score <= threshold, 0 (bona fide) if score > threshold
    predicted_labels = (all_scores <= eer_threshold).astype(int)
    
    # Calculate Accuracy at the EER threshold
    test_accuracy = accuracy_score(all_labels, predicted_labels)
    
    # Calculate Precision, Recall, F1 for the 'spoof' class (pos_label=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        predicted_labels, 
        average='binary', 
        pos_label=1,
        zero_division=0
    )
    
    # --- Logging and Printing ---
    log_msg = (
        f"\n--- Test Results ---\n"
        f"Total Test Samples: {len(all_labels)}\n"
        f"Bona fide Samples: {len(bonafide_scores)}\n"
        f"Spoofed Samples: {len(spoof_scores)}\n"
        f"EER Threshold (at min(abs(FNR-FPR))): {eer_threshold:.4f}\n"
        f"----------------------------------------\n"
        f"EER: {test_eer:.2f}%\n"
        f"AUC (Area Under Curve): {test_auc:.4f}\n"
        f"Accuracy (at EER threshold): {test_accuracy:.4f}\n"
        f"--- Metrics for 'Spoof' class (pos_label=1) at EER threshold ---\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-Score: {f1:.4f}\n"
        f"----------------------------------------"
    )
    
    print(log_msg) # Print to console
    
    if logger:
        logger.info(log_msg) # Log to file
            
    return test_eer, test_auc, test_accuracy, precision, recall, f1
