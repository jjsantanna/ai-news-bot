"""Per-control and aggregate metrics for evaluating extractor predictions."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from compliance_extractor.eval.matcher import match
from compliance_extractor.schemas import EvidenceCols, GoldCols


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    hallucinations: int = 0
    n_predictions: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinations / self.n_predictions if self.n_predictions else 0.0


def evaluate(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    quote_match_threshold: float = 0.8,
) -> Metrics:
    """Aggregate metrics across all controls."""
    metrics = Metrics()
    gold_by_control: dict[str, list[str]] = {}
    for _, row in gold.iterrows():
        gold_by_control.setdefault(str(row[GoldCols.CONTROL_ID]), []).append(str(row[GoldCols.EXACT_QUOTE]))

    pred_by_control: dict[str, list[tuple[str, bool, bool]]] = {}
    for _, row in predictions.iterrows():
        cid = str(row[EvidenceCols.CONTROL_ID])
        quote = str(row[EvidenceCols.EVIDENCE_TEXT] or "")
        validated = bool(row[EvidenceCols.VALIDATED_EXISTS]) if not pd.isna(row[EvidenceCols.VALIDATED_EXISTS]) else False
        method = str(row[EvidenceCols.MATCH_METHOD] or "")
        is_hallucinated = bool(quote) and (method in {"none", "semantic"})
        pred_by_control.setdefault(cid, []).append((quote, validated, is_hallucinated))

    all_controls = set(gold_by_control) | set(pred_by_control)
    for cid in all_controls:
        gold_quotes = gold_by_control.get(cid, [])
        preds = pred_by_control.get(cid, [])
        metrics.n_predictions += len(preds)

        if not preds:
            metrics.fn += len(gold_quotes)
            continue

        for quote, validated, hallucinated in preds:
            if hallucinated:
                metrics.hallucinations += 1
            if not gold_quotes:
                if quote and validated:
                    metrics.fp += 1
                continue
            matched = any(match(quote, gq, threshold=quote_match_threshold).is_match for gq in gold_quotes)
            if matched and validated:
                metrics.tp += 1
            elif quote and validated:
                metrics.fp += 1
            else:
                metrics.fn += 1

        n_pred_in_gold = sum(
            1 for (q, v, _) in preds if v and any(match(q, gq, threshold=quote_match_threshold).is_match for gq in gold_quotes)
        )
        if gold_quotes and n_pred_in_gold == 0 and not any(v for (_, v, _) in preds):
            metrics.fn += len(gold_quotes)
    return metrics


def metrics_to_row(variant_id: str, metrics: Metrics) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "hallucination_rate": metrics.hallucination_rate,
    }
