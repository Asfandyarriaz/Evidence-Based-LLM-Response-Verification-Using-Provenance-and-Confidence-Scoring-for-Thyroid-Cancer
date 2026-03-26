# rag/qa_pipeline.py
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder

try:
    from .faithfulness_evaluator import FaithfulnessEvaluator
    from .credibility_evaluator import CredibilityEvaluator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from faithfulness_evaluator import FaithfulnessEvaluator
    from credibility_evaluator import CredibilityEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVIDENCE_LEVEL_WEIGHTS: Dict[int, Tuple[str, float]] = {
    1: ("Guidelines / Consensus", 1.00),
    2: ("Systematic Review / Meta-analysis", 0.90),
    3: ("Randomized Controlled Trials", 0.80),
    4: ("Clinical Trials (non-randomized)", 0.70),
    5: ("Cohort Studies", 0.60),
    6: ("Case-Control Studies", 0.50),
    7: ("Case Reports / Series", 0.40),
}

FIRST_STAGE_RETRIEVAL  = 100
SECOND_STAGE_TOP_K     = 20
MAX_SOURCES            = 10
MAX_CHUNKS_PER_SOURCE  = 3
MAX_EXCERPT_CHARS      = 1200
MAX_TOTAL_CONTEXT_CHARS = 10000


class QAPipeline:
    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        llm_client: Any,
        instruction_file: str = "instructions/rag_instructions.txt",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.llm          = llm_client

        env  = os.getenv("ENV", "local").lower()
        base = Path(__file__).parent if env == "prod" else Path(".")
        self.instructions = (base / instruction_file).read_text(encoding="utf-8")

        logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        logger.info("Cross-encoder loaded successfully")

        logger.info("Initializing faithfulness evaluator")
        self.faithfulness_evaluator = FaithfulnessEvaluator(self.llm)
        logger.info("Faithfulness evaluator initialized")

        logger.info("Initializing credibility evaluator")
        self.credibility_evaluator = CredibilityEvaluator(
            pipeline=self,
            llm=self.llm,
        )
        logger.info("Credibility evaluator initialized")

        # Unanswerable questions for M5 — populated once at app startup
        # via pipeline.credibility_evaluator.unanswerable_cache
        # Set by app.py after init: pipeline.credibility_evaluator.unanswerable_cache = [...]
        self.credibility_evaluator.unanswerable_cache: Optional[List[str]] = None

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def _classify_question_type(self, question: str) -> str:
        pre_check = self._keyword_preclassify(question)
        if pre_check:
            logger.info(f"Pre-classified by keyword: {pre_check}")
            return pre_check

        classification_prompt = f"""Classify this thyroid cancer question into ONE category.

Categories:
- definition:          "What is X?", "Tell me about X", "Explain X", "Describe X"
- complications:       "What are complications/risks/side effects of X?"
- comparison:          "X vs Y", "compare X and Y", "how does X differ from Y",
                       "what characterises X compared to Y"
- treatment:           "How to treat X?", "What treatment options?", "How is X treated?"
- diagnosis:           "How is X diagnosed?", "What tests for X?", "Role of [tool] in X?"
- timing:              "When should X?", "When is X recommended?", "When to do X?"
- evidence:            "What is the evidence for X?", "Efficacy of X?",
                       "What trials support X?", "What outcomes does X show?"
- staging:             "What is the staging for X?", "AJCC/TNM for X?",
                       "How is X staged?", "What are the stages of X?"
- risk_stratification: "What defines low/high risk X?", "Risk categories for X?",
                       "What factors stratify risk in X?"
- impact:              "How does X affect Y?", "What is the impact of X on Y?",
                       "How does X influence management/prognosis?"
- surveillance:        "What surveillance after X?", "How is X monitored after treatment?",
                       "What follow-up tests after X?"
- recurrence:          "What are patterns of recurrence?", "How is recurrence detected?",
                       "How does X recur?"
- molecular:           "What is the role of [gene] in X?", "What is BRAF/RET/TERT?",
                       "Molecular drivers of X?"

Question: {question}

Critical rules:
- "evidence", "efficacy", "trial", "outcomes" keywords → evidence
- Gene names (BRAF, RET, TERT, NTRK, RAS) → molecular
- Staging or TNM → staging
- Low/intermediate/high risk comparison → risk_stratification
- "how does X affect management/prognosis" → impact
- Follow-up or surveillance after treatment → surveillance
- Recurrence patterns or detection → recurrence

Return ONLY the category name (one word), nothing else:"""

        try:
            category = self.llm.ask(classification_prompt).strip().lower()
            valid_categories = [
                "definition", "complications", "comparison", "treatment",
                "diagnosis", "timing", "evidence", "staging",
                "risk_stratification", "impact", "surveillance",
                "recurrence", "molecular"
            ]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}', defaulting to definition")
                category = "definition"

            if category == "definition":
                category = self._reclassify_if_diagnostic_tool(question, category)
            if category == "definition":
                category = self._reclassify_if_evidence_question(question, category)
            if category == "definition":
                category = self._reclassify_if_molecular_question(question, category)

            logger.info(f"Classified as: {category}")
            return category

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "definition"

    def _keyword_preclassify(self, question: str) -> Optional[str]:
        q = question.lower().strip()

        molecular_genes = [
            "braf", "ret", "tert", "ntrk", "ras", "nras", "kras", "hras",
            "pax8", "pparg", "tp53", "pten", "dicer1", "akt1", "v600e", "m918t",
        ]
        molecular_phrases = [
            "role of braf", "role of ret", "role of tert", "role of ntrk",
            "what is braf", "what is ret", "what is ras",
            "molecular driver", "oncogenic mutation", "germline mutation", "somatic mutation",
        ]
        if any(phrase in q for phrase in molecular_phrases):
            return "molecular"
        if any(f" {gene} " in f" {q} " for gene in molecular_genes):
            if any(w in q for w in ["role", "mutation", "what is", "how does", "relate", "driver", "target"]):
                return "molecular"

        staging_phrases = [
            "ajcc", "tnm", "staging", "how is thyroid cancer staged",
            "what are the stages", "stage i", "stage ii", "stage iii", "stage iv",
        ]
        if any(phrase in q for phrase in staging_phrases):
            return "staging"

        risk_phrases = [
            "low-risk vs", "low risk vs", "define low-risk", "define low risk",
            "risk categories", "risk stratification", "risk stratify",
            "what factors define", "what defines low",
            "intermediate risk", "high-risk differentiated",
            "ata risk", "recurrence risk category",
        ]
        if any(phrase in q for phrase in risk_phrases):
            return "risk_stratification"

        impact_phrases = [
            "how does lymph node", "how does nodal", "how does metastasis affect",
            "how does invasion affect", "how does extension affect",
            "impact of lymph", "impact of nodal", "impact of metastasis",
            "effect of lymph node", "affect management", "affect prognosis",
            "affect survival", "influence management", "influence prognosis",
        ]
        if any(phrase in q for phrase in impact_phrases):
            return "impact"

        surveillance_phrases = [
            "surveillance after", "follow-up after", "follow up after",
            "monitoring after", "tests after treatment", "imaging after",
            "what tests are used for surveillance", "what imaging is used for surveillance",
            "post-treatment surveillance", "post treatment monitoring",
        ]
        if any(phrase in q for phrase in surveillance_phrases):
            return "surveillance"

        recurrence_phrases = [
            "patterns of recurrence", "pattern of recurrence",
            "how is recurrence detected", "how does thyroid cancer recur",
            "recurrence in differentiated", "sites of recurrence",
            "locoregional recurrence", "common recurrence",
        ]
        if any(phrase in q for phrase in recurrence_phrases):
            return "recurrence"

        evidence_phrases = [
            "what is the evidence", "what is the clinical evidence",
            "what evidence exists", "what does the evidence",
            "what does research show", "what do studies show",
            "what clinical trials", "what trials exist", "what trials support",
            "evidence for", "evidence of", "evidence supporting", "evidence base for",
            "is there evidence", "what are the outcomes of", "what are clinical outcomes",
            "efficacy of", "how effective is", "how efficacious",
            "what is the efficacy", "clinical evidence for", "trial data", "trial evidence",
        ]
        if any(phrase in q for phrase in evidence_phrases):
            return "evidence"

        complication_phrases = [
            "what are the complications", "what are complications",
            "what are the risks of", "what are risks of",
            "what are the side effects", "what are side effects",
            "what are adverse", "side effects of", "risks of",
            "complications of", "adverse effects of", "adverse events of",
        ]
        if any(phrase in q for phrase in complication_phrases):
            return "complications"

        comparison_phrases = [
            " vs ", " versus ", "difference between", "compare ",
            "how does management differ", "how does it differ",
            "what characterises", "characterizes and how",
        ]
        if any(phrase in q for phrase in comparison_phrases):
            return "comparison"

        timing_starters = [
            "when should", "when is", "when to ", "when do ", "when are", "when would",
        ]
        if any(q.startswith(p) or f" {p}" in q for p in timing_starters):
            return "timing"

        return None

    def _reclassify_if_diagnostic_tool(self, question: str, original_category: str) -> str:
        q = question.lower()
        diagnostic_tools = [
            "ultrasound", "ultrasonography", "sonography", "fnab", "fna",
            "fine-needle", "fine needle", "biopsy", "imaging", "scan", "ct scan",
            "mri", "pet scan", "elastography", "doppler", "thyroglobulin",
            "calcitonin", "tsh", "biomarker", "molecular testing", "genetic testing",
            "laryngoscopy", "endoscopy", "ti-rads", "tirads", "acr",
        ]
        role_indicators = [
            "role of", "role in", "use of", "use in", "used for", "used in",
            "purpose of", "utility of", "value of", "how is", "how does", "how can",
            "evaluate", "evaluating", "evaluation", "assess", "assessing",
            "detect", "detecting", "detection",
        ]
        mentions_tool = any(tool in q for tool in diagnostic_tools)
        mentions_role = any(ind in q for ind in role_indicators)
        if mentions_tool and (mentions_role or "what is" in q or "what does" in q):
            logger.info(f"Reclassifying '{question}' → diagnosis (diagnostic tool)")
            return "diagnosis"
        return original_category

    def _reclassify_if_evidence_question(self, question: str, original_category: str) -> str:
        q = question.lower()
        strong_phrases = [
            "what is the evidence", "evidence for", "evidence of", "evidence on",
            "is there evidence", "what trials", "efficacy of",
            "how effective is", "what studies", "clinical evidence", "trial data",
        ]
        if any(phrase in q for phrase in strong_phrases):
            return "evidence"
        weak = [
            "evidence", "trial", "study", "studies", "research", "efficacy",
            "effective", "outcome", "outcomes", "survival", "response rate",
            "pfs", "os", "randomized", "systematic review", "meta-analysis",
            "phase 3", "phase 2", "rct",
        ]
        if sum(1 for w in weak if w in q) >= 2:
            return "evidence"
        return original_category

    def _reclassify_if_molecular_question(self, question: str, original_category: str) -> str:
        q = question.lower()
        genes = ["braf", "ret", "tert", "ntrk", "ras", "nras", "kras", "pten", "tp53"]
        molecular_context = [
            "mutation", "gene", "oncogene", "proto-oncogene", "molecular",
            "pathway", "mapk", "driver", "germline", "somatic", "hereditary",
        ]
        mentions_gene    = any(gene in q for gene in genes)
        mentions_context = any(ctx in q for ctx in molecular_context)
        if mentions_gene and mentions_context:
            return "molecular"
        if mentions_gene and any(w in q for w in ["role", "relate", "what is", "how does"]):
            return "molecular"
        return original_category

    # =========================================================================
    # RETRIEVAL HELPERS
    # =========================================================================

    def _rerank_with_cross_encoder(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: int = SECOND_STAGE_TOP_K
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        logger.info(f"Re-ranking {len(chunks)} chunks with cross-encoder...")
        pairs = []
        for chunk in chunks:
            doc_text = chunk.get("text", "")
            title    = chunk.get("title", "")
            combined = f"{title}. {doc_text}" if title and title not in doc_text else doc_text
            pairs.append([question, combined[:2000]])
        scores = self.cross_encoder.predict(pairs)
        for idx, chunk in enumerate(chunks):
            chunk["score"] = float(scores[idx])
        reranked   = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        top_chunks = reranked[:top_k]
        logger.info(f"Re-ranking complete: top={top_chunks[0]['score']:.4f}")
        for i, chunk in enumerate(top_chunks[:3], 1):
            logger.info(f"  Rank {i}: {chunk['score']:.4f} — {chunk.get('title','N/A')[:60]}")
        return top_chunks

    def diagnose_retrieval(self, question: str, k: int = 10) -> Dict[str, Any]:
        logger.info("=== DIAGNOSTIC MODE ===")
        sub_queries = self._expand_query_with_llm(question)
        diagnosis   = {
            "original_question":    question,
            "sub_queries_generated": sub_queries,
            "retrieval_results":    [],
        }
        for idx, sub_query in enumerate(sub_queries, 1):
            chunks = self.vector_store.search(sub_query, k=k)
            result = {"query": sub_query, "chunks_found": len(chunks), "sample_chunks": []}
            for i, chunk in enumerate(chunks[:3], 1):
                result["sample_chunks"].append({
                    "rank":           i,
                    "title":          chunk.get("title", "No title"),
                    "year":           chunk.get("year", "Unknown"),
                    "pmid":           chunk.get("pmid", "Unknown"),
                    "evidence_level": chunk.get("evidence_level", "Unknown"),
                    "score":          chunk.get("score", 0.0),
                    "text_preview":   chunk.get("text", "")[:400] + "...",
                })
            diagnosis["retrieval_results"].append(result)
        logger.info("=== END DIAGNOSTIC ===")
        return diagnosis

    def _expand_query_with_llm(self, question: str) -> List[str]:
        expansion_prompt = f"""You are a medical information retrieval assistant specialised in thyroid cancer.
Given a user question, generate 3-5 targeted search queries to retrieve comprehensive information.

Guidelines:
1. Include the original question
2. Focus on: main topic, complications/risks, clinical outcomes, patient selection, alternatives
3. Use specific medical terminology from research papers

Question: {question}

Return ONLY a JSON array of 3-5 search queries, no other text:"""
        try:
            response = self.llm.ask(expansion_prompt)
            cleaned  = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            queries = json.loads(cleaned)
            if isinstance(queries, list) and len(queries) > 0:
                if question not in queries:
                    queries.insert(0, question)
                queries = self._add_fallback_queries(question, queries)
                logger.info(f"Expanded to {len(queries)} queries")
                return queries
            return self._create_fallback_queries(question)
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return self._create_fallback_queries(question)

    def _add_fallback_queries(self, original_question: str, existing_queries: List[str]) -> List[str]:
        q          = original_question.lower()
        additional = []

        if any(w in q for w in ["evidence", "trial", "efficacy", "outcome", "study", "data", "effective"]):
            topic = self._extract_topic(original_question) or "thyroid cancer treatment"
            additional.extend([
                f"randomized controlled trial {topic}",
                f"phase 3 trial {topic} progression-free survival",
                f"clinical outcomes {topic} thyroid cancer",
                f"FDA approval {topic} thyroid cancer",
                f"real-world data {topic} thyroid cancer",
            ])
        elif any(w in q for w in ["complication", "risk", "side effect", "adverse", "toxicity"]):
            topic = self._extract_topic(original_question)
            if topic:
                additional.extend([
                    f"adverse effects {topic}", f"toxicity {topic}", f"late complications {topic}",
                ])
        elif any(w in q for w in ["recurrence", "recur"]):
            additional.extend([
                "locoregional recurrence differentiated thyroid cancer",
                "thyroglobulin recurrence detection",
                "neck ultrasound recurrence surveillance",
            ])
        elif any(w in q for w in ["surveillance", "follow-up", "monitoring"]):
            additional.extend([
                "thyroglobulin monitoring thyroid cancer follow-up",
                "neck ultrasound surveillance differentiated thyroid cancer",
                "radioiodine scan follow-up thyroid cancer",
            ])
        elif any(w in q for w in ["braf", "ret", "molecular", "mutation", "tert", "ntrk"]):
            additional.extend([
                "BRAF V600E papillary thyroid cancer prognosis",
                "RET mutation medullary thyroid cancer targeted therapy",
                "molecular targeted therapy thyroid cancer",
            ])

        if any(term in q for term in ["radioactive iodine", "rai", "i-131", "radioiodine"]):
            additional.extend([
                "salivary gland dysfunction radioactive iodine",
                "xerostomia RAI therapy",
                "secondary malignancy radioiodine",
            ])

        for query in additional:
            if query not in existing_queries:
                existing_queries.append(query)
        return existing_queries

    def _extract_topic(self, question: str) -> Optional[str]:
        q = question.lower()
        for pattern in [
            r'(?:of|for)\s+(.+?)(?:\?|$)',
            r'(?:what|how)\s+(?:is|are)\s+(.+?)(?:\?|treated|diagnosed)',
        ]:
            match = re.search(pattern, q)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'\s+(in|for|with)\s+thyroid.*', ' thyroid', topic)
                return topic
        return None

    def _create_fallback_queries(self, question: str) -> List[str]:
        q       = question.lower()
        queries = [question]
        if any(w in q for w in ["complication", "risk", "adverse", "side effect"]):
            topic = self._extract_topic(question) or "thyroid cancer treatment"
            queries.extend([
                f"complications {topic}", f"adverse effects {topic}", f"toxicity {topic}",
            ])
        elif any(w in q for w in ["surgical", "surgery", "procedure", "operation"]):
            queries.extend([
                f"{question.replace('?','')} complications",
                f"{question.replace('?','')} risks",
            ])
        return queries

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen   = set()
        unique = []
        for chunk in chunks:
            chunk_id = f"{chunk.get('pmid','unknown')}||{hash(chunk.get('text','')[:200].strip())}"
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)
        logger.info(f"Deduplicated {len(chunks)} → {len(unique)} chunks")
        return unique

    def _build_tagged_context(
        self, retrieved: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Dict]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title"))
            grouped.setdefault(key, []).append(r)

        source_map  = {}
        tagged_parts = []
        total_chars  = 0
        source_id    = 1

        for group in grouped.values():
            if source_id > MAX_SOURCES:
                break
            meta       = group[0]
            source_tag = f"SOURCE_{source_id}"
            source_map[source_tag] = {
                "title":               meta.get("title", "Unknown"),
                "year":                meta.get("year", "Unknown"),
                "pmid":                meta.get("pmid", "Unknown"),
                "evidence_level":      meta.get("evidence_level", "Unknown"),
                "cross_encoder_score": meta.get("score", 0.0),
            }
            for chunk in group[:MAX_CHUNKS_PER_SOURCE]:
                text        = chunk.get("text", "")[:MAX_EXCERPT_CHARS]
                tagged_text = f"[{source_tag}] {text}"
                if total_chars + len(tagged_text) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                tagged_parts.append(tagged_text)
                total_chars += len(tagged_text)
            source_id += 1

        context = "\n\n".join(tagged_parts)
        logger.info(f"Built context with {len(source_map)} sources")
        return context, source_map

    def _compute_confidence(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        levels = [
            r.get("evidence_level") for r in retrieved
            if r.get("evidence_level") in EVIDENCE_LEVEL_WEIGHTS
        ]
        if not levels:
            return {"label": "Low", "score": 0, "breakdown": "No evidence metadata"}
        weights    = [EVIDENCE_LEVEL_WEIGHTS[l][1] for l in levels]
        score      = int(round((sum(weights) / len(weights)) * 100))
        label      = "High" if score >= 85 else "Medium" if score >= 65 else "Low"
        breakdown  = "; ".join(
            f"Level {l} ({EVIDENCE_LEVEL_WEIGHTS[l][0]}): {levels.count(l)}"
            for l in sorted(set(levels))
        )
        return {"label": label, "score": score, "breakdown": breakdown}

    # =========================================================================
    # PROMPT TEMPLATES — 13 question types
    # =========================================================================

    def _create_type_specific_prompt(self, question: str, context: str, question_type: str) -> str:
        base = f"""
{self.instructions}

You are a medical information assistant specialised in thyroid cancer.
Answer by SYNTHESISING the tagged excerpts below. Every factual claim must cite
the excerpt(s) it came from using [SOURCE_X] tags.

SOURCE TAGGING RULES:
- Every factual sentence must end with one or more source tags: "Fact. [SOURCE_1]"
- Combine tags for multi-source synthesis: "Fact. [SOURCE_1][SOURCE_3]"
- Never write a factual sentence without at least one [SOURCE_X] tag.
- When you paraphrase or synthesise across excerpts, cite all contributing sources.

SYNTHESIS CONTRACT:
1. SYNTHESISE from the excerpts — combine, paraphrase, and infer where the excerpts
   collectively support a claim. You do not need verbatim text; reasonable medical
   synthesis from the excerpts is encouraged.
2. Set a field to null ONLY if NO excerpt in the context addresses that topic at all.
   Do not set fields to null simply because the phrasing in the excerpts is indirect.
3. NEVER invent facts that cannot be traced to at least one excerpt. If a specific
   number, drug name, or criterion is not present in any excerpt, do not include it.
4. NEVER write placeholder text like "[Topic]", "[Drug Name]", "Name", "Type Name".
5. ALL string values must be real synthesised sentences — not template examples.
6. Return ONLY valid JSON. No markdown fences, no preamble, no trailing text.
7. Missing sections that cannot be addressed from ANY excerpt: set content/items to
   null silently. Do NOT explain what is missing.

CONTEXT WITH SOURCE TAGS:
{context}

"""

        if question_type == "evidence":
            return base + f"""
QUESTION: {question}

Synthesise evidence across all sources. Organise by cancer subtype, not by source.
Extract trial names and numerical outcomes from the excerpts wherever they appear.

CONTRACT ADDITIONS:
- Drug items MUST include trial name + numerical outcome if present in any excerpt.
- Key Considerations section REQUIRED, minimum 2 items synthesised from excerpts.
- Items arrays must remain arrays even with one item.
- If only partial trial data exists in excerpts, synthesise what is there and cite it.

{{
  "overview": "2-3 sentences on TKI role, approval status, overall evidence quality synthesised from excerpts. [SOURCE_X]. REQUIRED — never null.",
  "sections": [
    {{
      "header": "Evidence in Differentiated Thyroid Cancer (DTC)",
      "items": [
        {{
          "name": "Exact drug name from excerpts. null if no DTC drug appears in any excerpt.",
          "description": "Trial name + primary endpoint with exact numbers if in excerpts, otherwise synthesise available DTC evidence. [SOURCE_X]. null only if no DTC evidence exists in any excerpt.",
          "highlight": "Clinical significance synthesised from excerpts. [SOURCE_X]. null only if excerpts contain nothing relevant."
        }}
      ]
    }},
    {{
      "header": "Evidence in Medullary Thyroid Cancer (MTC)",
      "items": [
        {{
          "name": "Exact drug name from excerpts. null if no MTC drug appears in any excerpt.",
          "description": "Trial name + endpoint numbers if available; otherwise synthesise available MTC evidence from excerpts. [SOURCE_X]. null only if no MTC evidence exists.",
          "highlight": "Clinical positioning synthesised from excerpts. [SOURCE_X]. null only if no relevant MTC content."
        }}
      ]
    }},
    {{
      "header": "Evidence in Anaplastic Thyroid Cancer (ATC)",
      "items": [
        {{
          "name": "Exact drug or combination from excerpts. null if no ATC data in any excerpt.",
          "description": "Evidence strength and key outcomes synthesised from excerpts. [SOURCE_X]. null only if no ATC evidence.",
          "highlight": "Key limitation or rationale synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Toxicity Profile",
          "description": "Synthesise adverse event information from excerpts, including percentages if stated. [SOURCE_X]. null only if no toxicity data in any excerpt."
        }},
        {{
          "consideration": "Resistance",
          "description": "Synthesise any resistance or salvage information from excerpts. [SOURCE_X]. null only if excerpts contain nothing on resistance."
        }},
        {{
          "consideration": "Patient Selection",
          "description": "Synthesise patient selection criteria from excerpts (e.g. radioiodine-refractory, progressive disease). [SOURCE_X]. null only if no selection data in excerpts."
        }},
        {{
          "consideration": "Treatment Sequencing",
          "description": "Synthesise sequencing guidance from excerpts. [SOURCE_X]. null only if no sequencing information in any excerpt."
        }}
      ]
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "definition":
            return base + f"""
QUESTION: {question}

Synthesise a clear, patient-facing definition from the excerpts. Even if individual
excerpts are narrow or specialist, combine them to construct a coherent overview.

{{
  "overview": "2-3 sentence definition synthesised from excerpts: what thyroid cancer is, who it affects, general outlook. [SOURCE_X]. REQUIRED — never null.",
  "sections": [
    {{
      "header": "Key Types",
      "items": [
        {{
          "name": "Actual type name from excerpts. null only if no type information appears in any excerpt.",
          "description": "What distinguishes this type, incidence if mentioned, synthesised from excerpts. [SOURCE_X]. null only if no type data.",
          "details": "Prognosis or statistics from excerpts. [SOURCE_X]. null only if no prognostic data in any excerpt."
        }}
      ]
    }},
    {{
      "header": "Key Characteristics",
      "items": [
        {{
          "aspect": "Real aspect synthesised from excerpts. null only if excerpts contain no characteristic information.",
          "description": "Explanation synthesised from excerpts. [SOURCE_X]. null only if no relevant excerpt content."
        }}
      ]
    }},
    {{
      "header": "Causes and Risk Factors",
      "content": "Paragraph synthesising causes and risk factors from excerpts. [SOURCE_X]. null only if no excerpt mentions any cause or risk factor."
    }},
    {{
      "header": "Common Symptoms",
      "items": [
        {{
          "symptom": "Actual symptom from excerpts. null only if no symptom information appears in any excerpt.",
          "description": "How it presents, synthesised from excerpts. [SOURCE_X]. null only if no symptom content."
        }}
      ]
    }},
    {{
      "header": "Diagnosis and Treatment",
      "content": "Diagnostic and treatment approach synthesised from excerpts. [SOURCE_X]. null only if no diagnostic or treatment content in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "complications":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentence summary of the complication landscape synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Common and Temporary Complications",
      "items": [
        {{
          "complication": "Real complication name from excerpts. null only if no common complications mentioned.",
          "description": "Explanation, mechanism, frequency synthesised from excerpts. [SOURCE_X]. null only if no relevant content.",
          "frequency": "Exact percentage from excerpts if stated; otherwise null."
        }}
      ]
    }},
    {{
      "header": "Less Common or Potentially Permanent Complications",
      "items": [
        {{
          "complication": "Real complication name from excerpts. null only if no such complications mentioned.",
          "description": "Explanation and long-term impact synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Rare but Serious Complications",
      "items": [
        {{
          "complication": "Real complication name from excerpts. null only if no rare complications mentioned.",
          "description": "Why serious and how managed, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Management and Prevention",
      "content": "Managing or preventing complications, synthesised from excerpts. [SOURCE_X]. null only if no management content in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "comparison":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentence comparison summary with clinical context synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Key Differences",
      "option_a_label": "Exact name of first option from question.",
      "option_b_label": "Exact name of second option from question.",
      "comparison_table": [
        {{
          "aspect": "Specific aspect synthesised from excerpts. null only if no comparison data exists.",
          "option_a": "Synthesised description for first option. [SOURCE_X]. null only if no excerpt covers this.",
          "option_b": "Synthesised description for second option. [SOURCE_X]. null only if no excerpt covers this."
        }}
      ]
    }},
    {{
      "header": "Characteristics of First Option",
      "items": [
        {{
          "aspect": "Real characteristic from excerpts. null only if no characteristics mentioned.",
          "description": "Synthesised explanation. [SOURCE_X]. null only if no relevant excerpt."
        }}
      ]
    }},
    {{
      "header": "Management Steps",
      "steps": [
        {{
          "step": 1,
          "title": "Real step title synthesised from excerpts. null only if no steps mentioned.",
          "description": "What this step involves, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Shared Characteristics",
      "content": "What both options have in common, synthesised from excerpts. [SOURCE_X]. null only if no shared features mentioned."
    }},
    {{
      "header": "Clinical Outcomes",
      "content": "Comparing survival, response rates with numbers if available, synthesised from excerpts. [SOURCE_X]. null only if no outcome data in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "treatment":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentence treatment landscape synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "First-Line Treatments",
      "items": [
        {{
          "treatment": "Real treatment name from excerpts. null only if no first-line treatment mentioned.",
          "description": "Indication, mechanism, trial result with numbers if in excerpts, approval status if mentioned. [SOURCE_X]. null only if no relevant content.",
          "evidence_grade": "Phase 3 RCT / Guideline / Observational — synthesised from excerpts. null if not determinable."
        }}
      ]
    }},
    {{
      "header": "Second-Line and Salvage Treatments",
      "items": [
        {{
          "treatment": "Real treatment name from excerpts. null only if no second-line treatment mentioned.",
          "description": "When used, outcomes, trial data synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Treatment by Cancer Subtype",
      "content": "Mapping approved agents to DTC, MTC, ATC subtypes, synthesised from excerpts. [SOURCE_X]. null only if no subtype-treatment mapping in any excerpt."
    }},
    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Real consideration from excerpts. null only if no considerations mentioned.",
          "description": "Concise synthesised explanation with any data. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "diagnosis":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentence summary of the diagnostic approach and its clinical value, synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Key Roles",
      "items": [
        {{
          "role": "Real role from excerpts. null only if no role information in any excerpt.",
          "description": "What this role involves and its clinical value, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Key Diagnostic Features or Findings",
      "items": [
        {{
          "feature": "Real feature from excerpts. null only if no features mentioned.",
          "description": "Clinical significance, specificity/sensitivity if stated in excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Diagnostic Pathway",
      "content": "Step-by-step process from presentation to diagnosis, synthesised from excerpts. [SOURCE_X]. null only if no pathway information in any excerpt."
    }},
    {{
      "header": "Limitations and Considerations",
      "content": "Known limitations of the diagnostic approach, synthesised from excerpts. [SOURCE_X]. null only if no limitations mentioned in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "timing":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentence summary of when and why this is recommended, synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Key Indications",
      "items": [
        {{
          "indication": "Real clinical situation from excerpts. null only if no indications in any excerpt.",
          "explanation": "Why this timing is recommended, with supporting data synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Size and Risk Thresholds",
      "items": [
        {{
          "category": "Real risk/size category from excerpts. null only if no thresholds in any excerpt.",
          "threshold": "Specific size or criterion from excerpts. null only if not stated in any excerpt.",
          "recommendation": "What is recommended at this threshold, synthesised from excerpts. [SOURCE_X]. null only if no recommendation content."
        }}
      ]
    }},
    {{
      "header": "Important Considerations",
      "items": [
        {{
          "consideration": "Real factor from excerpts. null only if no considerations in any excerpt.",
          "description": "How this factor influences the timing decision, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Contraindications or When Not Recommended",
      "content": "Situations where this should be avoided, synthesised from excerpts. [SOURCE_X]. null only if no contraindication information in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "staging":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: staging system, cancer type it applies to, key distinguishing features synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "TNM Components",
      "items": [
        {{
          "component": "T, N, or M — synthesised from excerpts.",
          "description": "What this component measures, synthesised from excerpts. [SOURCE_X]. null only if no TNM content in any excerpt.",
          "categories": "Exact subcategories from excerpts. null only if no subcategories stated."
        }}
      ]
    }},
    {{
      "header": "Stage Groupings",
      "subgroups": [
        {{
          "subgroup": "Patient subgroup from excerpts. null only if no groupings in any excerpt.",
          "stages": [
            {{
              "stage": "Stage label from excerpts. null only if not in any excerpt.",
              "criteria": "TNM criteria synthesised from excerpts. [SOURCE_X]. null only if not stated."
            }}
          ]
        }}
      ]
    }},
    {{
      "header": "Clinical vs Pathologic Staging",
      "content": "cTNM vs pTNM distinction if mentioned in excerpts. [SOURCE_X]. null only if not addressed in any excerpt."
    }},
    {{
      "header": "Key Considerations",
      "content": "Important caveats or clinical implications synthesised from excerpts. [SOURCE_X]. null only if no such content in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "risk_stratification":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: stratification system, what it predicts, how it guides treatment — synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Low Risk",
      "recurrence_rate": "Exact percentage range from excerpts. null only if not stated in any excerpt.",
      "criteria": [
        {{
          "criterion": "Real criterion synthesised from excerpts. null only if no low-risk criteria in any excerpt.",
          "detail": "Clarification synthesised from excerpts. [SOURCE_X]. null only if no supporting detail."
        }}
      ]
    }},
    {{
      "header": "Intermediate Risk",
      "recurrence_rate": "Exact percentage range from excerpts. null only if not stated.",
      "criteria": [
        {{
          "criterion": "Real criterion synthesised from excerpts. null only if no intermediate criteria in any excerpt.",
          "detail": "Clarification synthesised from excerpts. [SOURCE_X]. null only if no supporting detail."
        }}
      ]
    }},
    {{
      "header": "High Risk",
      "recurrence_rate": "Exact percentage range from excerpts. null only if not stated.",
      "criteria": [
        {{
          "criterion": "Real criterion synthesised from excerpts. null only if no high-risk criteria in any excerpt.",
          "detail": "Clarification synthesised from excerpts. [SOURCE_X]. null only if no supporting detail."
        }}
      ]
    }},
    {{
      "header": "Dynamic Risk Restratification",
      "content": "How risk category can change based on treatment response, synthesised from excerpts. [SOURCE_X]. null only if not addressed in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "impact":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: how common is this factor, its primary clinical impact, overall significance — synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Impact on Management",
      "items": [
        {{
          "aspect": "Real management aspect from excerpts. null only if no management impact in any excerpt.",
          "description": "How this factor changes management, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Impact on Prognosis",
      "items": [
        {{
          "aspect": "Real prognostic aspect from excerpts. null only if no prognostic data in any excerpt.",
          "description": "Specific impact with data synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Age-Related Differences",
      "content": "How impact differs by patient age, synthesised from excerpts. [SOURCE_X]. null only if no age-related content in any excerpt."
    }},
    {{
      "header": "Summary of Impact",
      "table": [
        {{
          "domain": "Domain from excerpts. null only if no domain data.",
          "impact": "One-sentence synthesised impact. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "surveillance":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: what surveillance consists of, primary tests, risk-stratified approach — synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Key Surveillance Modalities",
      "items": [
        {{
          "modality": "Real test name from excerpts. null only if no modality information in any excerpt.",
          "description": "What it detects, when used, why preferred — synthesised from excerpts. [SOURCE_X]. null only if no relevant content.",
          "frequency": "How often performed if stated in any excerpt. null only if no frequency data."
        }}
      ]
    }},
    {{
      "header": "Surveillance Strategy by Risk Group",
      "items": [
        {{
          "risk_group": "Real risk group from excerpts. null only if no risk stratification in any excerpt.",
          "approach": "Recommended surveillance approach synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Special Considerations",
      "content": "Tg antibody interference, stimulated Tg testing, or other nuances synthesised from excerpts. [SOURCE_X]. null only if no special considerations in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "recurrence":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: overall recurrence rate, most common site, primary detection method — synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Common Patterns of Recurrence",
      "items": [
        {{
          "pattern": "Real location from excerpts. null only if no recurrence patterns in any excerpt.",
          "description": "Where exactly, percentage of recurrences, synthesised from excerpts. [SOURCE_X]. null only if no relevant content.",
          "frequency": "Percentage from excerpts if stated. null only if not in any excerpt."
        }}
      ]
    }},
    {{
      "header": "Methods of Detection",
      "items": [
        {{
          "method": "Real detection method from excerpts. null only if no detection methods in any excerpt.",
          "description": "How it detects recurrence, when used — synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Risk Factors for Recurrence",
      "items": [
        {{
          "factor": "Real risk factor from excerpts. null only if no risk factors in any excerpt.",
          "description": "Why this increases recurrence risk, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Clinical Signs of Recurrence",
      "content": "Symptoms or signs of recurrence synthesised from excerpts. [SOURCE_X]. null only if no clinical signs in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        elif question_type == "molecular":
            return base + f"""
QUESTION: {question}

{{
  "overview": "2-3 sentences: what this gene/mutation is, how common it is, overall clinical importance — synthesised from excerpts. [SOURCE_X]. REQUIRED.",
  "sections": [
    {{
      "header": "Mechanism of Action",
      "content": "How this mutation drives cancer growth and which pathway it activates, synthesised from excerpts. [SOURCE_X]. null only if no mechanism information in any excerpt."
    }},
    {{
      "header": "Prevalence and Cancer Type Association",
      "items": [
        {{
          "cancer_type": "Real cancer type from excerpts. null only if no type association in any excerpt.",
          "prevalence": "Exact percentage from excerpts. null only if not stated in any excerpt.",
          "mutation_subtype": "Specific mutation subtype from excerpts. null only if not in any excerpt.",
          "description": "Clinical characteristics associated with this mutation, synthesised from excerpts. [SOURCE_X]. null only if no relevant content."
        }}
      ]
    }},
    {{
      "header": "Prognostic Significance",
      "content": "Whether mutation predicts aggressive behaviour, recurrence, or survival — synthesised from excerpts. [SOURCE_X]. null only if no prognostic data in any excerpt."
    }},
    {{
      "header": "Therapeutic Implications",
      "items": [
        {{
          "therapy": "Real targeted therapy from excerpts. null only if no therapy mentioned in any excerpt.",
          "indication": "What mutation this targets, synthesised from excerpts. [SOURCE_X]. null only if no indication content.",
          "outcome": "Key outcome data synthesised from excerpts. [SOURCE_X]. null only if no outcome data."
        }}
      ]
    }},
    {{
      "header": "Genetic Testing Guidance",
      "content": "When to test, method used, clinical decisions it informs — synthesised from excerpts. [SOURCE_X]. null only if no testing guidance in any excerpt."
    }}
  ]
}}
Return ONLY valid JSON:"""

        else:
            logger.warning(f"Unknown type '{question_type}', falling back to definition")
            return self._create_type_specific_prompt(question, context, "definition")

    # =========================================================================
    # MAIN ANSWER METHOD
    # =========================================================================

    def answer(
        self,
        question: str,
        chat_history: Optional[list] = None,
        k: int = 30,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1.  Classify
          2.  Expand queries
          3.  Bi-encoder retrieval
          4.  Deduplicate
          5.  Cross-encoder reranking
          6.  Build tagged context
          7.  Compute evidence confidence
          8.  Generate JSON answer
          9.  Faithfulness evaluation
          10. Credibility scorecard (all 7 metrics)
        """
        # 1
        question_type = self._classify_question_type(question)

        # 2
        sub_queries = self._expand_query_with_llm(question)

        # 3
        logger.info("=== FIRST STAGE: Bi-encoder retrieval ===")
        all_retrieved    = []
        chunks_per_query = FIRST_STAGE_RETRIEVAL // len(sub_queries)
        for idx, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Sub-query {idx}/{len(sub_queries)}: {sub_query}")
            retrieved = self.vector_store.search(sub_query, k=chunks_per_query)
            all_retrieved.extend(retrieved)
        logger.info(f"First stage: {len(all_retrieved)} chunks")

        # 4
        unique_retrieved = self._deduplicate_chunks(all_retrieved)
        if not unique_retrieved:
            return {
                "error":      "No relevant information found",
                "json_response": None,
                "sources":    {},
                "confidence": {"label": "Low", "score": 0, "breakdown": "No data"},
            }

        # 5
        logger.info("=== SECOND STAGE: Cross-encoder re-ranking ===")
        reranked_chunks = self._rerank_with_cross_encoder(
            question=question, chunks=unique_retrieved, top_k=SECOND_STAGE_TOP_K
        )

        # 6
        context, source_map = self._build_tagged_context(reranked_chunks)

        # 7
        confidence = self._compute_confidence(reranked_chunks)

        # 8
        logger.info(f"Generating '{question_type}' answer...")
        prompt = self._create_type_specific_prompt(question, context, question_type)
        try:
            response = self.llm.ask(prompt).strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*\n?', '', response)
                response = re.sub(r'\n?```\s*$', '', response)
            json_response = json.loads(response)

            # 9 — Faithfulness
            logger.info("Evaluating faithfulness...")
            try:
                faithfulness = self.faithfulness_evaluator.evaluate(
                    json_response=json_response,
                    tagged_context=context,
                    source_map=source_map,
                )
                logger.info(
                    f"Faithfulness: {faithfulness.get('label','N/A')} "
                    f"({faithfulness.get('score','N/A')})"
                )
            except Exception as e:
                logger.error(f"Faithfulness evaluation failed: {e}", exc_info=True)
                faithfulness = {
                    "score": None, "label": "Not Available",
                    "error": str(e), "total_statements": 0, "evaluated_statements": 0,
                }

            # Partial result so credibility evaluator can read it
            partial_result = {
                "json_response":   json_response,
                "sources":         source_map,
                "confidence":      confidence,
                "faithfulness":    faithfulness,
                "question_type":   question_type,
                "retrieval_stats": {
                    "first_stage_retrieved": len(all_retrieved),
                    "after_dedup":           len(unique_retrieved),
                    "after_reranking":       len(reranked_chunks),
                },
            }

            # 10 — Credibility scorecard (all 7 metrics)
            logger.info("Computing credibility scorecard (all 7 metrics)...")
            try:
                credibility_scorecard = self.credibility_evaluator.compute_scorecard(
                    question=question,
                    result=partial_result,
                    run_consistency=True,
                    unanswerable_questions=self.credibility_evaluator.unanswerable_cache,
                )
                logger.info(
                    f"Credibility score: {credibility_scorecard.get('total_score','N/A')}/100 "
                    f"({credibility_scorecard.get('overall_label','N/A')})"
                )
            except Exception as e:
                logger.error(f"Credibility scorecard failed: {e}", exc_info=True)
                credibility_scorecard = {
                    "error":         str(e),
                    "total_score":   None,
                    "overall_label": "Not Available",
                }

            return {
                **partial_result,
                "credibility_scorecard": credibility_scorecard,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nResponse was: {response}")
            return {
                "error":         f"Failed to generate structured response: {str(e)}",
                "json_response": None,
                "sources":       source_map,
                "confidence":    confidence,
            }
