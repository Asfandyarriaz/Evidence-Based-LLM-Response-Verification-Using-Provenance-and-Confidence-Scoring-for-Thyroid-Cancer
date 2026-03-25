# rag/credibility_evaluator.py
import re
import csv
import json
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Evidence level → trust score proxy for Metric 1
EVIDENCE_TRUST_SCORES: Dict[int, float] = {
    1: 1.00,   # Guidelines / Consensus
    2: 0.90,   # Systematic Review / Meta-analysis
    3: 0.80,   # Randomized Controlled Trials
    4: 0.70,   # Clinical Trials (non-randomized)
    5: 0.60,   # Cohort Studies
    6: 0.50,   # Case-Control Studies
    7: 0.40,   # Case Reports / Series
}

EVIDENCE_LEVEL_NAMES: Dict[int, str] = {
    1: "Guidelines / Consensus",
    2: "Systematic Review / Meta-analysis",
    3: "Randomized Controlled Trials",
    4: "Clinical Trials (non-randomized)",
    5: "Cohort Studies",
    6: "Case-Control Studies",
    7: "Case Reports / Series",
}

# Scoring weights (must sum to 100)
METRIC_WEIGHTS: Dict[str, int] = {
    "factual_correctness":    30,
    "grounding":              20,
    "hallucination":          20,
    "confidence_calibration": 10,
    "safe_refusal":           10,
    "consistency":             5,
    "citation_quality":        5,
}


class CredibilityEvaluator:
    """
    Computes the 7-metric LLM Credibility Scorecard (0–100) for the
    thyroid cancer QA pipeline.

    Metric breakdown
    ────────────────
    M1  Factual Correctness     (30pts) — evidence-level trust proxy
    M2  Grounding               (20pts) — supported-claim rate
    M3  Hallucination Rate      (20pts) — unsupported-claim penalty
    M4  Confidence Calibration  (10pts) — |evidence_confidence – faithfulness|
    M5  Safe Refusal            (10pts) — correct null/refusal on unanswerable Qs
    M6  Consistency              (5pts) — answer stability across LLM paraphrases
    M7  Citation Quality         (5pts) — valid citation + PMID + CE-score quality
    """

    def __init__(
        self,
        pipeline: Any,
        llm: Any,
        results_dir: str = "eval/results",
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # M1 — Factual Correctness
    # =========================================================================

    def compute_factual_correctness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proxy: weighted average trust score of cited sources by evidence level.

        Rationale: Level 1 (clinical guidelines) is ground-truth by definition
        in an evidence-based medical system. Trust degrades as evidence level
        descends toward case reports (Level 7).
        """
        sources      = result.get("sources", {})
        json_response = result.get("json_response") or {}

        if not sources or not json_response:
            return {
                "score": 0.0, "score_0_100": 0.0,
                "label": "N/A", "detail": "No sources available",
            }

        answer_text    = json.dumps(json_response)
        cited_keys     = set(re.findall(r'SOURCE_\d+', answer_text))
        trust_scores   = []
        cited_details  = []

        for key in cited_keys:
            if key not in sources:
                continue
            level = sources[key].get("evidence_level")
            if isinstance(level, int) and level in EVIDENCE_TRUST_SCORES:
                trust = EVIDENCE_TRUST_SCORES[level]
                trust_scores.append(trust)
                cited_details.append({
                    "source":         key,
                    "evidence_level": level,
                    "level_name":     EVIDENCE_LEVEL_NAMES.get(level, "Unknown"),
                    "trust_score":    trust,
                    "title":          sources[key].get("title", "Unknown")[:60],
                })

        if not trust_scores:
            return {
                "score": 0.5, "score_0_100": 50.0,
                "label": "Medium",
                "detail": "No evidence-level metadata on cited sources",
            }

        avg_trust    = sum(trust_scores) / len(trust_scores)
        score_0_100  = round(avg_trust * 100, 1)
        label        = "High" if avg_trust >= 0.85 else "Medium" if avg_trust >= 0.65 else "Low"

        return {
            "score":              round(avg_trust, 3),
            "score_0_100":        score_0_100,
            "label":              label,
            "cited_sources":      len(cited_keys),
            "avg_trust_score":    round(avg_trust, 3),
            "detail":             cited_details,
        }

    # =========================================================================
    # M2 + M3 — Grounding & Hallucination
    # =========================================================================

    def compute_grounding_and_hallucination(
        self, faithfulness: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Grounding     = supported-claim rate (faithfulness score, 0–1).
        Hallucination = 1 – grounding rate.

        Both are derived directly from the existing FaithfulnessEvaluator
        output already computed in the main pipeline.
        """
        if not faithfulness or faithfulness.get("score") is None:
            return {
                "grounding_score":       None,
                "grounding_score_0_100": None,
                "grounding_label":       "N/A",
                "hallucination_rate":    None,
                "hallucination_rate_pct": None,
                "hallucination_label":   "N/A",
                "total_statements":      0,
                "evaluated_statements":  0,
            }

        grounding    = faithfulness["score"]
        hallucination = round(1.0 - grounding, 3)

        g_label = "High"   if grounding    >= 0.80 else "Medium" if grounding    >= 0.60 else "Low"
        h_label = "Low"    if hallucination <= 0.20 else "Medium" if hallucination <= 0.40 else "High"

        return {
            "grounding_score":        round(grounding, 3),
            "grounding_score_0_100":  round(grounding * 100, 1),
            "grounding_label":        g_label,
            "hallucination_rate":     hallucination,
            "hallucination_rate_pct": round(hallucination * 100, 1),
            "hallucination_label":    h_label,
            "total_statements":       faithfulness.get("total_statements", 0),
            "evaluated_statements":   faithfulness.get("evaluated_statements", 0),
        }

    # =========================================================================
    # M4 — Confidence Calibration
    # =========================================================================

    def compute_confidence_calibration(
        self,
        confidence:   Dict[str, Any],
        faithfulness: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calibration error (ECE proxy) = |normalised_confidence – faithfulness|.

        Pipeline confidence is 0–100 (evidence-level weighted).
        Faithfulness is 0–1.
        Score = 1 – ECE  (higher = better calibrated).

        Interpretation: a perfectly calibrated system that returns
        confidence=80 should produce faithfulness≈0.80. When they diverge
        it means the evidence quality signal and the LLM behaviour are
        de-coupled (e.g. high-quality sources but hallucinated answer).
        """
        if not confidence or not faithfulness:
            return {"ece": None, "score": None, "score_0_100": None, "label": "N/A"}

        conf_raw   = confidence.get("score")
        faith_raw  = faithfulness.get("score")

        if conf_raw is None or faith_raw is None:
            return {"ece": None, "score": None, "score_0_100": None, "label": "N/A"}

        conf_norm   = conf_raw / 100.0
        ece         = abs(conf_norm - faith_raw)
        calib_score = round(1.0 - ece, 3)

        label = (
            "Well-calibrated"    if ece <= 0.15 else
            "Moderate"           if ece <= 0.30 else
            "Poorly calibrated"
        )

        return {
            "confidence_input":   conf_raw,
            "faithfulness_input": round(faith_raw * 100, 1),
            "ece":                round(ece, 3),
            "ece_pct":            round(ece * 100, 1),
            "score":              calib_score,
            "score_0_100":        round(calib_score * 100, 1),
            "label":              label,
        }

    # =========================================================================
    # M5 — Safe Refusal / Uncertainty Handling
    # =========================================================================

    def generate_unanswerable_questions(self, n: int = 10) -> List[str]:
        """
        Ask LLM to generate questions that a thyroid cancer PubMed corpus
        should NOT be able to answer.

        Includes: unrelated medical topics, non-medical topics, future events,
        patient-specific questions, and implausibly niche queries.
        """
        prompt = f"""You are evaluating a thyroid cancer QA system backed by a PubMed database.
Generate exactly {n} questions that this system should NOT be able to answer.

Include a mix of:
- Unrelated medical topics (cardiology, orthopaedics, dermatology)
- Non-medical topics (finance, travel, sport, cooking)
- Future hypothetical events with no published evidence yet
- Questions requiring personal patient records
- Extremely niche edge cases with no PubMed literature

Rules:
- Do NOT include anything plausibly answerable from thyroid cancer literature
- Make them realistic — questions a user might actually type
- Vary question style (what/how/when/should/is)

Return ONLY a valid JSON array of {n} question strings. No preamble, no markdown:"""

        try:
            response = self.llm.ask(prompt).strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*\n?', '', response)
                response = re.sub(r'\n?```\s*$', '', response)
            questions = json.loads(response)
            if isinstance(questions, list) and questions:
                return [str(q) for q in questions[:n]]
        except Exception as e:
            logger.error(f"Unanswerable question generation failed: {e}")

        # Hardcoded fallback
        return [
            "What is the best treatment for acute myocardial infarction?",
            "How do I train for a marathon in 12 weeks?",
            "What will AI look like in 2050?",
            "How is rheumatoid arthritis diagnosed?",
            "What are the visa requirements for visiting Japan?",
            "How does quantum computing work?",
            "What is the cure for Alzheimer's disease discovered in 2031?",
            "What are the symptoms of scurvy?",
            "How do I fix a leaking roof?",
            "What is the best diet for weight loss?",
        ]

    def compute_safe_refusal(
        self, unanswerable_questions: List[str]
    ) -> Dict[str, Any]:
        """
        For each unanswerable question, run the pipeline and check whether
        the system correctly refuses / returns mostly null sections.

        Correct refusal criteria (either):
        - null_rate >= 0.60  (majority of JSON fields are null)
        - faithfulness < 0.35 (very low grounding — likely hallucinated)
        """
        results         = []
        correct_count   = 0

        for question in unanswerable_questions:
            try:
                result        = self.pipeline.answer(question)
                json_resp     = result.get("json_response") or {}
                faithfulness  = result.get("faithfulness", {})
                faith_score   = faithfulness.get("score")

                null_count, total_count = self._count_null_fields(json_resp)
                null_rate = null_count / total_count if total_count > 0 else 1.0

                is_correct = (
                    null_rate >= 0.60 or
                    (faith_score is not None and faith_score < 0.35)
                )
                if is_correct:
                    correct_count += 1

                results.append({
                    "question":          question,
                    "null_rate":         round(null_rate, 2),
                    "null_sections":     null_count,
                    "total_fields":      total_count,
                    "faithfulness":      faith_score,
                    "correct_refusal":   is_correct,
                })

            except Exception as e:
                logger.error(f"Safe refusal test error on '{question}': {e}")
                results.append({
                    "question":        question,
                    "error":           str(e),
                    "correct_refusal": False,
                })

        total        = len(unanswerable_questions)
        refusal_rate = correct_count / total if total > 0 else 0.0
        label        = "Good" if refusal_rate >= 0.75 else "Moderate" if refusal_rate >= 0.50 else "Poor"

        return {
            "correct_refusals":   correct_count,
            "total_tested":       total,
            "refusal_rate":       round(refusal_rate, 3),
            "refusal_rate_pct":   round(refusal_rate * 100, 1),
            "score":              round(refusal_rate, 3),
            "score_0_100":        round(refusal_rate * 100, 1),
            "label":              label,
            "per_question":       results,
        }

    def _count_null_fields(self, json_response: Dict) -> Tuple[int, int]:
        """Recursively count null vs populated string fields in the JSON response."""
        null_count  = 0
        total_count = 0

        def walk(obj: Any) -> None:
            nonlocal null_count, total_count
            if isinstance(obj, dict):
                for v in obj.values():
                    if v is None:
                        null_count  += 1
                        total_count += 1
                    elif isinstance(v, str) and v.strip():
                        total_count += 1
                    elif isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(json_response)
        return null_count, total_count

    # =========================================================================
    # M6 — Consistency / Stability
    # =========================================================================

    def compute_consistency(
        self,
        question: str,
        result:   Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reuses the pipeline's _expand_query_with_llm to get paraphrase variants
        of the original question (already generated during retrieval expansion).
        Runs up to 3 paraphrases through the pipeline and measures:

        Agreement = 0.4 × source_jaccard + 0.6 × LLM_semantic_similarity
        """
        try:
            paraphrases = self.pipeline._expand_query_with_llm(question)
            alts = [
                q for q in paraphrases
                if q.lower().strip() != question.lower().strip()
            ][:3]

            if not alts:
                return {
                    "score": None, "score_0_100": None,
                    "label": "N/A", "detail": "No distinct paraphrases generated",
                }

            orig_sources  = set(result.get("sources", {}).keys())
            orig_overview = (result.get("json_response") or {}).get("overview", "")

            agreement_scores = []
            per_paraphrase   = []

            for alt_q in alts:
                try:
                    alt_result   = self.pipeline.answer(alt_q)
                    alt_sources  = set(alt_result.get("sources", {}).keys())
                    alt_overview = (alt_result.get("json_response") or {}).get("overview", "")

                    # Jaccard similarity on retrieved source sets
                    union = orig_sources | alt_sources
                    jaccard = (
                        len(orig_sources & alt_sources) / len(union)
                        if union else 0.0
                    )

                    # LLM semantic similarity on overviews
                    semantic = self._llm_semantic_similarity(orig_overview, alt_overview)

                    agreement = round((jaccard * 0.4) + (semantic * 0.6), 3)
                    agreement_scores.append(agreement)

                    per_paraphrase.append({
                        "paraphrase":         alt_q,
                        "source_jaccard":     round(jaccard, 3),
                        "semantic_similarity": round(semantic, 3),
                        "agreement":          agreement,
                    })

                except Exception as e:
                    logger.error(f"Consistency sub-run failed for '{alt_q}': {e}")

            if not agreement_scores:
                return {
                    "score": None, "score_0_100": None,
                    "label": "N/A", "detail": "All paraphrase runs failed",
                }

            avg  = sum(agreement_scores) / len(agreement_scores)
            label = "Stable" if avg >= 0.75 else "Moderate" if avg >= 0.50 else "Unstable"

            return {
                "score":              round(avg, 3),
                "score_0_100":        round(avg * 100, 1),
                "label":              label,
                "paraphrases_tested": len(agreement_scores),
                "per_paraphrase":     per_paraphrase,
            }

        except Exception as e:
            logger.error(f"Consistency computation failed: {e}")
            return {"score": None, "score_0_100": None, "label": "N/A", "detail": str(e)}

    def _llm_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """LLM-judged semantic similarity between two answer overviews (0–1)."""
        if not text_a or not text_b:
            return 0.0

        clean_a = re.sub(r'\[SOURCE_\d+\]', '', text_a).strip()[:400]
        clean_b = re.sub(r'\[SOURCE_\d+\]', '', text_b).strip()[:400]

        prompt = f"""Score the semantic similarity of these two medical summaries.

Summary A: {clean_a}

Summary B: {clean_b}

Scoring guide:
1.0  = identical meaning
0.75 = mostly same, minor differences
0.5  = partial overlap, some different focus
0.25 = related topic, different content
0.0  = completely different

Respond with ONLY a single decimal number (e.g. 0.8). Nothing else:"""

        try:
            response = self.llm.ask(prompt).strip()
            match    = re.search(r'\d+\.?\d*', response)
            if match:
                return min(1.0, max(0.0, float(match.group())))
        except Exception as e:
            logger.error(f"LLM semantic similarity error: {e}")
        return 0.5

    # =========================================================================
    # M7 — Citation / Source Credibility
    # =========================================================================

    def compute_citation_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Three sub-checks:
        1. Valid citation rate     — [SOURCE_X] tags exist in source_map (weight 0.50)
        2. PMID validity rate      — cited sources have real PMIDs    (weight 0.25)
        3. Cross-encoder relevance — cited sources have high CE score  (weight 0.25)
        """
        sources       = result.get("sources", {})
        json_response = result.get("json_response") or {}
        answer_text   = json.dumps(json_response)

        cited_tags   = set(re.findall(r'SOURCE_\d+', answer_text))
        total_cites  = len(cited_tags)

        if total_cites == 0:
            return {
                "total_source_tags": 0,
                "valid_citation_rate": 0.0,
                "pmid_validity_rate":  0.0,
                "score":       0.0,
                "score_0_100": 0.0,
                "label":       "No citations",
                "detail":      "No [SOURCE_X] tags found in answer",
            }

        # 1 — valid citations
        valid = sum(1 for t in cited_tags if t in sources)
        valid_rate = valid / total_cites

        # 2 — PMID validity
        pmid_valid = sum(
            1 for t in cited_tags
            if t in sources and sources[t].get("pmid") not in (None, "", "Unknown")
        )
        pmid_rate = pmid_valid / total_cites

        # 3 — cross-encoder relevance of cited sources
        ce_scores = [
            sources[t].get("cross_encoder_score", 0.0)
            for t in cited_tags if t in sources
        ]
        avg_ce = sum(ce_scores) / len(ce_scores) if ce_scores else 0.0
        # ms-marco-MiniLM scores typically range −10 → +10; normalise to 0–1
        ce_norm = min(1.0, max(0.0, (avg_ce + 10.0) / 20.0))

        combined = (valid_rate * 0.50) + (pmid_rate * 0.25) + (ce_norm * 0.25)
        label    = "High" if combined >= 0.80 else "Medium" if combined >= 0.55 else "Low"

        return {
            "total_source_tags":     total_cites,
            "valid_citations":       valid,
            "valid_citation_rate":   round(valid_rate, 3),
            "valid_citation_pct":    round(valid_rate * 100, 1),
            "pmid_valid_count":      pmid_valid,
            "pmid_validity_rate":    round(pmid_rate, 3),
            "pmid_validity_pct":     round(pmid_rate * 100, 1),
            "avg_cross_encoder":     round(avg_ce, 4),
            "ce_normalised":         round(ce_norm, 3),
            "score":                 round(combined, 3),
            "score_0_100":           round(combined * 100, 1),
            "label":                 label,
        }

    # =========================================================================
    # COMBINED SCORECARD
    # =========================================================================

    def compute_scorecard(
        self,
        question:                str,
        result:                  Dict[str, Any],
        run_consistency:         bool                = True,
        unanswerable_questions:  Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all 7 metrics and combine into a 0–100 credibility score.

        Args:
            question:               Original question string
            result:                 Output from pipeline.answer()
            run_consistency:        Whether to run M6 (adds ~3 pipeline calls)
            unanswerable_questions: Pre-generated list for M5; if None M5 is skipped
        """
        logger.info(f"Computing scorecard for: '{question[:70]}'")

        faithfulness = result.get("faithfulness", {})
        confidence   = result.get("confidence", {})

        m1   = self.compute_factual_correctness(result)
        m2m3 = self.compute_grounding_and_hallucination(faithfulness)
        m4   = self.compute_confidence_calibration(confidence, faithfulness)

        m5 = (
            self.compute_safe_refusal(unanswerable_questions)
            if unanswerable_questions
            else {"score": None, "score_0_100": None, "label": "Not run", "refusal_rate_pct": None}
        )

        m6 = (
            self.compute_consistency(question, result)
            if run_consistency
            else {"score": None, "score_0_100": None, "label": "Not run"}
        )

        m7 = self.compute_citation_quality(result)

        # ── Weighted total score ──────────────────────────────────────────────
        contributions     = {}
        total_weight_used = 0

        def add(name: str, score_0_100: Optional[float], weight: int) -> None:
            nonlocal total_weight_used
            if score_0_100 is not None:
                contributions[name] = {
                    "score":        round(score_0_100, 1),
                    "weight":       weight,
                    "contribution": round(score_0_100 * weight / 100, 2),
                }
                total_weight_used += weight

        add("factual_correctness",   m1.get("score_0_100"),                  METRIC_WEIGHTS["factual_correctness"])
        add("grounding",             m2m3.get("grounding_score_0_100"),       METRIC_WEIGHTS["grounding"])

        # Hallucination is a penalty: contribution = (100 − hall_pct) × weight / 100
        hall_pct = m2m3.get("hallucination_rate_pct")
        if hall_pct is not None:
            inv = 100.0 - hall_pct
            contributions["hallucination_penalty"] = {
                "score":        round(inv, 1),
                "weight":       METRIC_WEIGHTS["hallucination"],
                "contribution": round(inv * METRIC_WEIGHTS["hallucination"] / 100, 2),
            }
            total_weight_used += METRIC_WEIGHTS["hallucination"]

        add("confidence_calibration", m4.get("score_0_100"),  METRIC_WEIGHTS["confidence_calibration"])
        add("safe_refusal",           m5.get("score_0_100"),  METRIC_WEIGHTS["safe_refusal"])
        add("consistency",            m6.get("score_0_100"),  METRIC_WEIGHTS["consistency"])
        add("citation_quality",       m7.get("score_0_100"),  METRIC_WEIGHTS["citation_quality"])

        raw_total = sum(v["contribution"] for v in contributions.values())

        # If some metrics were skipped, rescale to available weight
        if 0 < total_weight_used < 100:
            total_score = round(raw_total * 100 / total_weight_used, 1)
        else:
            total_score = round(raw_total, 1)

        total_score   = min(100.0, max(0.0, total_score))
        overall_label = (
            "Excellent" if total_score >= 85 else
            "Good"      if total_score >= 70 else
            "Fair"      if total_score >= 55 else
            "Poor"
        )

        return {
            "question":          question,
            "timestamp":         datetime.datetime.utcnow().isoformat(),
            "question_type":     result.get("question_type", "unknown"),
            "total_score":       total_score,
            "overall_label":     overall_label,
            "metrics": {
                "m1_factual_correctness":    m1,
                "m2_grounding": {
                    "score":               m2m3.get("grounding_score"),
                    "score_0_100":         m2m3.get("grounding_score_0_100"),
                    "label":               m2m3.get("grounding_label"),
                    "total_statements":    m2m3.get("total_statements"),
                    "evaluated":           m2m3.get("evaluated_statements"),
                },
                "m3_hallucination": {
                    "rate":     m2m3.get("hallucination_rate"),
                    "rate_pct": m2m3.get("hallucination_rate_pct"),
                    "label":    m2m3.get("hallucination_label"),
                },
                "m4_confidence_calibration": m4,
                "m5_safe_refusal":           m5,
                "m6_consistency":            m6,
                "m7_citation_quality":       m7,
            },
            "weighted_breakdown":  contributions,
            "total_weight_used":   total_weight_used,
            "retrieval_stats":     result.get("retrieval_stats", {}),
        }

    # =========================================================================
    # BATCH EVALUATION
    # =========================================================================

    def run_batch_evaluation(
        self,
        questions:         List[str],
        run_consistency:   bool = False,
        n_unanswerable:    int  = 10,
    ) -> Dict[str, Any]:
        """
        Run the full scorecard over a list of test questions.

        - Generates unanswerable questions once (M5) and reuses them per card.
        - Saves full JSON + summary CSV to self.results_dir.
        - run_consistency=False by default because it triples LLM calls.
          Set True for a thorough periodic evaluation.
        """
        logger.info(f"Batch evaluation started: {len(questions)} questions")

        logger.info("Generating unanswerable questions for M5...")
        unanswerable = self.generate_unanswerable_questions(n=n_unanswerable)

        all_scorecards: List[Dict] = []

        for i, q in enumerate(questions, 1):
            logger.info(f"  [{i}/{len(questions)}] {q[:70]}")
            try:
                result    = self.pipeline.answer(q)
                scorecard = self.compute_scorecard(
                    question=q,
                    result=result,
                    run_consistency=run_consistency,
                    unanswerable_questions=unanswerable,
                )
                all_scorecards.append(scorecard)
            except Exception as e:
                logger.error(f"Failed for '{q}': {e}")
                all_scorecards.append({
                    "question":    q,
                    "error":       str(e),
                    "total_score": None,
                })

        valid_scores = [
            s["total_score"] for s in all_scorecards
            if s.get("total_score") is not None
        ]

        summary = {
            "total_questions":        len(questions),
            "successful_evaluations": len(valid_scores),
            "avg_total_score":        round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None,
            "min_score":              min(valid_scores) if valid_scores else None,
            "max_score":              max(valid_scores) if valid_scores else None,
            "per_metric_averages":    self._aggregate_metrics(all_scorecards),
        }

        batch_result = {
            "timestamp":                    datetime.datetime.utcnow().isoformat(),
            "summary":                      summary,
            "scorecards":                   all_scorecards,
            "unanswerable_questions_used":  unanswerable,
        }

        json_path, csv_path = self._save_results(batch_result)
        batch_result["saved_json"] = json_path
        batch_result["saved_csv"]  = csv_path

        logger.info(f"Batch evaluation complete. Avg score: {summary['avg_total_score']}")
        return batch_result

    def _aggregate_metrics(self, scorecards: List[Dict]) -> Dict[str, Optional[float]]:
        """Average each metric's score_0_100 across all scorecards."""
        slots = [
            ("m1_factual_correctness",    "score_0_100"),
            ("m2_grounding",              "score_0_100"),
            ("m3_hallucination",          "rate_pct"),
            ("m4_confidence_calibration", "score_0_100"),
            ("m5_safe_refusal",           "score_0_100"),
            ("m6_consistency",            "score_0_100"),
            ("m7_citation_quality",       "score_0_100"),
        ]
        result = {}
        for key, field in slots:
            vals = [
                sc.get("metrics", {}).get(key, {}).get(field)
                for sc in scorecards
                if sc.get("metrics", {}).get(key, {}).get(field) is not None
            ]
            result[key] = round(sum(vals) / len(vals), 1) if vals else None
        return result

    def _save_results(self, batch: Dict) -> Tuple[str, str]:
        """Persist batch results to JSON (full) and CSV (summary per question)."""
        ts       = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = self.results_dir / f"eval_{ts}.json"
        csv_path  = self.results_dir / f"eval_{ts}.csv"

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, default=str)
        logger.info(f"Saved: {json_path}")

        # CSV
        try:
            fields = [
                "timestamp", "question", "question_type",
                "total_score", "overall_label",
                "m1_factual_correctness", "m2_grounding",
                "m3_hallucination_rate_pct",
                "m4_calibration", "m5_safe_refusal",
                "m6_consistency", "m7_citation_quality",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for sc in batch["scorecards"]:
                    m = sc.get("metrics", {})
                    writer.writerow({
                        "timestamp":               sc.get("timestamp", ""),
                        "question":                sc.get("question", "")[:120],
                        "question_type":           sc.get("question_type", ""),
                        "total_score":             sc.get("total_score", ""),
                        "overall_label":           sc.get("overall_label", ""),
                        "m1_factual_correctness":  m.get("m1_factual_correctness", {}).get("score_0_100", ""),
                        "m2_grounding":            m.get("m2_grounding", {}).get("score_0_100", ""),
                        "m3_hallucination_rate_pct": m.get("m3_hallucination", {}).get("rate_pct", ""),
                        "m4_calibration":          m.get("m4_confidence_calibration", {}).get("score_0_100", ""),
                        "m5_safe_refusal":         m.get("m5_safe_refusal", {}).get("score_0_100", ""),
                        "m6_consistency":          m.get("m6_consistency", {}).get("score_0_100", ""),
                        "m7_citation_quality":     m.get("m7_citation_quality", {}).get("score_0_100", ""),
                    })
            logger.info(f"Saved: {csv_path}")
        except Exception as e:
            logger.error(f"CSV save failed: {e}")

        return str(json_path), str(csv_path)
