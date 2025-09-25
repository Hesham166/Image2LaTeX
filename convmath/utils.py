import math
from typing import List
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute BLEU score (corpus-level).
    Args:
        references: list of reference strings (ground-truth LaTeX).
        hypotheses: list of predicted strings.
    Returns:
        BLEU score (0–100).
    """
    refs = [[r.split()] for r in references]
    hyps = [h.split() for h in hypotheses]
    smoothie = SmoothingFunction().method4
    score = corpus_bleu(refs, hyps, smoothing_function=smoothie)
    return score * 100.0


def compute_edit_distance(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute average normalized edit distance.
    Normalization = edit distance / max(len(ref, hyp)).
    Args:
        references: list of reference strings.
        hypotheses: list of predicted strings.
    Returns:
        Average normalized edit distance (0–1).
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        d = levenshtein_distance(" ".join(ref_tokens), " ".join(hyp_tokens))
        denom = max(len(ref_tokens), len(hyp_tokens), 1)
        scores.append(d / denom)
    return float(np.mean(scores))


def compute_exact_match(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute exact match accuracy.
    """
    matches = sum(r.strip() == h.strip() for r, h in zip(references, hypotheses))
    return matches / max(1, len(references))


@torch.no_grad()
def evaluate_metrics(vocab, model, loader, criterion, device="cuda"):
    """
    Evaluate loss, BLEU, edit distance, exact match on a dataloader.
    """
    model.eval()
    refs, hyps = [], []
    total_loss = 0
    num_samples = 0
    
    for imgs, tgts in tqdm(loader, desc="Evaluating"):
        if imgs.numel() == 0:
            continue
        imgs, tgts = imgs.to(device), tgts.to(device)
        
        # --- Loss Calculation ---
        logits = model(imgs, tgts[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))
        total_loss += loss.item() * imgs.size(0) # Weight loss by batch size
        num_samples += imgs.size(0)

        # --- Metric Calculation ---
        preds = model(imgs)  # greedy generation
        for pred, tgt in zip(preds, tgts):
            refs.append(vocab.decode(tgt.tolist()))
            hyps.append(vocab.decode(pred.tolist()))
            
    avg_loss = total_loss / max(1, num_samples)
    
    metrics = {
        "bleu": compute_bleu(refs, hyps),
        "edit_distance": compute_edit_distance(refs, hyps),
        "exact_match": compute_exact_match(refs, hyps),
    }
    
    return avg_loss, metrics