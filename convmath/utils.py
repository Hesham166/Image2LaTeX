import math
from typing import List
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from Levenshtein import distance as levenshtein_distance


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
def evaluate_metrics(vocab, model, loader, device="cuda"):
    """
    Evaluate BLEU, edit distance, exact match on a dataloader.
    Args:
        vocab: Vocab object with .decode()
        model: ConvMath model
        loader: DataLoader
        device: torch device
    Returns:
        dict with 'bleu', 'edit_distance', 'exact_match'
    """
    model.eval()
    refs, hyps = [], []
    for imgs, tgts in loader:
        if imgs.numel() == 0:
            continue
        imgs, tgts = imgs.to(device), tgts.to(device)
        preds = model(imgs)  # greedy generation
        for pred, tgt in zip(preds, tgts):
            refs.append(vocab.decode(tgt.tolist()))
            hyps.append(vocab.decode(pred.tolist()))
    return {
        "bleu": compute_bleu(refs, hyps),
        "edit_distance": compute_edit_distance(refs, hyps),
        "exact_match": compute_exact_match(refs, hyps),
    }
