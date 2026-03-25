# eval/run_evaluation.py
"""
Standalone batch evaluation script.

Usage:
    python eval/run_evaluation.py                          # default test set
    python eval/run_evaluation.py --questions my_qs.json  # custom question file
    python eval/run_evaluation.py --consistency           # include M6 (slow)
    python eval/run_evaluation.py --n-unanswerable 15     # more M5 questions

Results are saved to eval/results/ as JSON + CSV.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_runner")

# ── Default thyroid cancer test question set ──────────────────────────────────
DEFAULT_QUESTIONS = [
    # Definition
    "What is thyroid cancer?",
    "What is papillary thyroid cancer?",
    "What is medullary thyroid cancer?",
    "What is anaplastic thyroid cancer?",
    "What is follicular thyroid cancer?",

    # Diagnosis
    "What is the role of ultrasound in thyroid cancer diagnosis?",
    "How is thyroid cancer diagnosed?",
    "What is the role of fine-needle aspiration biopsy in thyroid nodule evaluation?",

    # Treatment
    "How is differentiated thyroid cancer treated?",
    "What is the role of radioactive iodine in thyroid cancer treatment?",
    "How is medullary thyroid cancer treated?",

    # Evidence
    "What is the evidence for lenvatinib in thyroid cancer?",
    "What is the evidence for sorafenib in differentiated thyroid cancer?",
    "What is the clinical evidence for vandetanib in medullary thyroid cancer?",

    # Staging
    "What is the AJCC staging system for differentiated thyroid cancer?",
    "How is thyroid cancer staged?",

    # Risk stratification
    "What defines low risk differentiated thyroid cancer?",
    "What are the ATA risk stratification categories for thyroid cancer?",

    # Surveillance
    "What surveillance is recommended after thyroid cancer treatment?",
    "What is the role of thyroglobulin in thyroid cancer surveillance?",

    # Recurrence
    "What are the patterns of recurrence in differentiated thyroid cancer?",
    "How is recurrence detected in thyroid cancer?",

    # Molecular
    "What is the role of BRAF mutation in papillary thyroid cancer?",
    "What is the role of RET mutation in medullary thyroid cancer?",
    "What is the significance of TERT promoter mutations in thyroid cancer?",

    # Complications
    "What are the complications of thyroidectomy?",
    "What are the side effects of radioactive iodine therapy?",

    # Impact
    "How does lymph node metastasis affect prognosis in thyroid cancer?",
    "How does BRAF V600E affect management of papillary thyroid cancer?",

    # Timing
    "When should fine needle aspiration be performed on a thyroid nodule?",
    "When is radioactive iodine indicated after thyroidectomy?",

    # Comparison
    "What is the difference between papillary and follicular thyroid cancer?",
    "How does differentiated thyroid cancer differ from anaplastic thyroid cancer?",
]


def print_scorecard_summary(scorecard: dict) -> None:
    """Pretty-print a single scorecard to stdout."""
    q      = scorecard.get("question", "")[:80]
    total  = scorecard.get("total_score")
    label  = scorecard.get("overall_label", "N/A")
    m      = scorecard.get("metrics", {})

    print(f"\n{'─'*72}")
    print(f"Q: {q}")
    print(f"{'─'*72}")
    if scorecard.get("error"):
        print(f"  ERROR: {scorecard['error']}")
        return

    print(f"  Total Credibility Score: {total}/100  ({label})")
    print()

    rows = [
        ("M1 Factual Correctness",   m.get("m1_factual_correctness", {}).get("score_0_100"),
                                     m.get("m1_factual_correctness", {}).get("label")),
        ("M2 Grounding",             m.get("m2_grounding", {}).get("score_0_100"),
                                     m.get("m2_grounding", {}).get("label")),
        ("M3 Hallucination Rate",    m.get("m3_hallucination", {}).get("rate_pct"),
                                     m.get("m3_hallucination", {}).get("label") + " hallucination"
                                     if m.get("m3_hallucination", {}).get("label") else ""),
        ("M4 Calibration",           m.get("m4_confidence_calibration", {}).get("score_0_100"),
                                     m.get("m4_confidence_calibration", {}).get("label")),
        ("M5 Safe Refusal",          m.get("m5_safe_refusal", {}).get("score_0_100"),
                                     m.get("m5_safe_refusal", {}).get("label")),
        ("M6 Consistency",           m.get("m6_consistency", {}).get("score_0_100"),
                                     m.get("m6_consistency", {}).get("label")),
        ("M7 Citation Quality",      m.get("m7_citation_quality", {}).get("score_0_100"),
                                     m.get("m7_citation_quality", {}).get("label")),
    ]

    for name, score, lbl in rows:
        score_str = f"{score:.1f}" if score is not None else "N/A"
        lbl_str   = f"({lbl})" if lbl else ""
        print(f"  {name:<30} {score_str:>6}  {lbl_str}")


def print_batch_summary(batch: dict) -> None:
    """Pretty-print the overall batch summary."""
    s = batch.get("summary", {})
    p = s.get("per_metric_averages", {})

    print(f"\n{'═'*72}")
    print("  BATCH EVALUATION SUMMARY")
    print(f"{'═'*72}")
    print(f"  Questions evaluated : {s.get('successful_evaluations')}/{s.get('total_questions')}")
    print(f"  Avg credibility     : {s.get('avg_total_score')}/100")
    print(f"  Min / Max           : {s.get('min_score')} / {s.get('max_score')}")
    print()
    print("  Per-metric averages:")
    metric_display = [
        ("M1 Factual Correctness",  "m1_factual_correctness"),
        ("M2 Grounding",            "m2_grounding"),
        ("M3 Hallucination Rate",   "m3_hallucination"),
        ("M4 Calibration",          "m4_confidence_calibration"),
        ("M5 Safe Refusal",         "m5_safe_refusal"),
        ("M6 Consistency",          "m6_consistency"),
        ("M7 Citation Quality",     "m7_citation_quality"),
    ]
    for display_name, key in metric_display:
        val = p.get(key)
        val_str = f"{val:.1f}" if val is not None else "Not run"
        print(f"    {display_name:<30} {val_str}")

    print()
    saved_json = batch.get("saved_json", "")
    saved_csv  = batch.get("saved_csv", "")
    if saved_json:
        print(f"  Full results → {saved_json}")
    if saved_csv:
        print(f"  CSV summary  → {saved_csv}")
    print(f"{'═'*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM Credibility Scorecard evaluation")
    parser.add_argument(
        "--questions", type=str, default=None,
        help="Path to JSON file containing list of question strings. "
             "Defaults to built-in thyroid cancer test set.",
    )
    parser.add_argument(
        "--consistency", action="store_true", default=False,
        help="Include M6 Consistency (adds ~3× pipeline calls per question — slow).",
    )
    parser.add_argument(
        "--n-unanswerable", type=int, default=10,
        help="Number of unanswerable questions to generate for M5 (default: 10).",
    )
    parser.add_argument(
        "--results-dir", type=str, default="eval/results",
        help="Directory to save JSON + CSV results (default: eval/results).",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print per-question scorecards to stdout.",
    )
    args = parser.parse_args()

    # Load questions
    if args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {args.questions}")
    else:
        questions = DEFAULT_QUESTIONS
        logger.info(f"Using default test set: {len(questions)} questions")

    # Initialise pipeline
    logger.info("Initialising pipeline...")
    try:
        from core.pipeline_loader import init_pipeline
        pipeline = init_pipeline()
    except Exception as e:
        logger.error(f"Pipeline initialisation failed: {e}")
        sys.exit(1)

    # Initialise evaluator
    from rag.credibility_evaluator import CredibilityEvaluator
    evaluator = CredibilityEvaluator(
        pipeline=pipeline,
        llm=pipeline.llm,
        results_dir=args.results_dir,
    )

    logger.info(
        f"Starting batch evaluation — consistency={'ON' if args.consistency else 'OFF'}, "
        f"n_unanswerable={args.n_unanswerable}"
    )

    batch = evaluator.run_batch_evaluation(
        questions=questions,
        run_consistency=args.consistency,
        n_unanswerable=args.n_unanswerable,
    )

    # Print output
    if args.verbose:
        for sc in batch["scorecards"]:
            print_scorecard_summary(sc)

    print_batch_summary(batch)


if __name__ == "__main__":
    main()
