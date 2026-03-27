# ui/json_renderer.py
import re
from typing import Dict, Any, List, Optional


class JSONRenderer:
    """
    Converts structured JSON responses to markdown with clickable source citations.
    Renders credibility scorecard (7 metrics) inside the Show Sources collapsible.
    """

    def __init__(
        self,
        json_response: Dict[str, Any],
        sources: Dict[str, Dict],
        confidence: Dict[str, Any],
        faithfulness: Dict[str, Any] = None,
        credibility_scorecard: Optional[Dict[str, Any]] = None,
    ):
        self.json_response         = json_response
        self.sources               = sources
        self.confidence            = confidence
        self.faithfulness          = faithfulness or {}
        self.credibility_scorecard = credibility_scorecard or {}
        self.citation_count        = 0

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _replace_source_tags(self, text: str) -> str:
        """Replace [SOURCE_X] tags with clickable anchor-linked citations."""
        if not text:
            return ""
        pattern = r'\[SOURCE_(\d+)\]'

        def replacement(match):
            source_num = match.group(1)
            source_key = f"SOURCE_{source_num}"
            if source_key in self.sources:
                source = self.sources[source_key]
                title  = source.get('title', 'Unknown')
                year   = source.get('year', '')
                return (
                    f'<a href="#source-{source_num}" '
                    f'class="citation-link" '
                    f'onclick="document.getElementById(\'sources-section\').open=true;" '
                    f'title="{title} ({year})">'
                    f'[{source_num}]</a>'
                )
            return f'[{source_num}]'

        return re.sub(pattern, replacement, text)

    def _clean_frequency(self, freq: str) -> str:
        if not freq:
            return ""
        unwanted_phrases = [
            "Not quantified in available excerpts", "Not specified",
            "Not mentioned", "Not quantified",
            "(Not specified)", "(Not quantified)",
        ]
        cleaned = freq
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if not cleaned or cleaned in ['()', '( )']:
            return ""
        return cleaned

    # =========================================================================
    # FAITHFULNESS (existing — unchanged)
    # =========================================================================

    def _render_faithfulness_badge(self) -> str:
        if not self.faithfulness or self.faithfulness.get("score") is None:
            error = self.faithfulness.get("error", "")
            if error:
                return '<div class="faithfulness-badge unavailable">⚠️ Faithfulness: Not Available</div>'
            return ""

        score      = self.faithfulness.get("score", 0)
        label      = self.faithfulness.get("label", "Unknown")
        percentage = int(score * 100)

        if score >= 0.80:
            css_class, icon = "high", "✅"
        elif score >= 0.60:
            css_class, icon = "medium", "⚠️"
        else:
            css_class, icon = "low", "❌"

        total     = self.faithfulness.get("total_statements", 0)
        evaluated = self.faithfulness.get("evaluated_statements", 0)
        tooltip   = f"{evaluated}/{total} statements verified"

        return f'''<div class="faithfulness-badge {css_class}" title="{tooltip}">
{icon} <strong>Faithfulness:</strong> {percentage}% ({label})
</div>'''

    def _render_faithfulness_for_sources(self) -> str:
        if not self.faithfulness or self.faithfulness.get("score") is None:
            error = self.faithfulness.get("error", "")
            if error:
                return '''
<div class="faithfulness-quality">
<strong>Answer Faithfulness:</strong> Not Available
<br>
<em>Evaluation could not be completed</em>
</div>'''
            return ""

        score      = self.faithfulness.get("score", 0)
        label      = self.faithfulness.get("label", "Unknown")
        percentage = int(score * 100)
        total      = self.faithfulness.get("total_statements", 0)
        evaluated  = self.faithfulness.get("evaluated_statements", 0)
        color_class = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"

        return f'''
<div class="faithfulness-quality {color_class}">
<strong>Answer Faithfulness:</strong> {percentage}% ({label})
<br>
<em>Based on: {evaluated}/{total} statements verified against sources</em>
</div>'''

    # =========================================================================
    # CREDIBILITY SCORECARD (NEW)
    # =========================================================================

    def _render_credibility_scorecard(self) -> str:
        """
        Render the 7-metric credibility scorecard as an HTML block inside
        the Show Sources collapsible, immediately after the faithfulness block.
        """
        sc = self.credibility_scorecard
        if not sc or sc.get("error"):
            err = sc.get("error", "") if sc else ""
            if err:
                return f'''
<div class="credibility-scorecard" style="margin-top:10px;padding:10px;
     border:0.5px solid var(--color-border-tertiary);border-radius:8px;">
<strong>Credibility Scorecard:</strong>
<span style="color:var(--color-text-secondary);font-size:13px;"> Not Available — {err[:120]}</span>
</div>'''
            return ""

        total = sc.get("total_score")
        label = sc.get("overall_label", "N/A")
        m     = sc.get("metrics", {})

        # Colour for total score
        if total is None:
            total_color, total_display = "var(--color-text-secondary)", "N/A"
        elif total >= 85:
            total_color, total_display = "#0F6E56", f"{total:.0f}"
        elif total >= 70:
            total_color, total_display = "#854F0B", f"{total:.0f}"
        elif total >= 55:
            total_color, total_display = "#BA7517", f"{total:.0f}"
        else:
            total_color, total_display = "#A32D2D", f"{total:.0f}"

        def pill(val, invert: bool = False) -> str:
            if val is None:
                return '<span style="color:var(--color-text-secondary);font-size:11px;">N/A</span>'
            v = (100 - val) if invert else val
            if v >= 80:
                bg, fg = "#EAF3DE", "#3B6D11"
            elif v >= 60:
                bg, fg = "#FAEEDA", "#854F0B"
            else:
                bg, fg = "#FCEBEB", "#A32D2D"
            return (
                f'<span style="background:{bg};color:{fg};font-size:11px;font-weight:500;'
                f'padding:2px 8px;border-radius:99px;white-space:nowrap;">{val:.0f}/100</span>'
            )

        def hall_pill(rate_pct) -> str:
            if rate_pct is None:
                return '<span style="color:var(--color-text-secondary);font-size:11px;">N/A</span>'
            if rate_pct <= 20:
                bg, fg = "#EAF3DE", "#3B6D11"
            elif rate_pct <= 40:
                bg, fg = "#FAEEDA", "#854F0B"
            else:
                bg, fg = "#FCEBEB", "#A32D2D"
            return (
                f'<span style="background:{bg};color:{fg};font-size:11px;font-weight:500;'
                f'padding:2px 8px;border-radius:99px;white-space:nowrap;">{rate_pct:.0f}%</span>'
            )

        m1 = m.get("m1_factual_correctness", {})
        m2 = m.get("m2_grounding", {})
        m3 = m.get("m3_hallucination", {})
        m4 = m.get("m4_confidence_calibration", {})
        m5 = m.get("m5_safe_refusal", {})
        m6 = m.get("m6_consistency", {})
        m7 = m.get("m7_citation_quality", {})

        rows_data = [
            ("M1", "Factual Correctness",  pill(m1.get("score_0_100")),
             m1.get("label",""), "Evidence-level trust proxy"),
            ("M2", "Grounding",            pill(m2.get("score_0_100")),
             m2.get("label",""), "Supported-claim rate"),
            ("M3", "Hallucination Rate",   hall_pill(m3.get("rate_pct")),
             m3.get("label",""), "% unsupported claims — lower is better"),
            ("M4", "Calibration",          pill(m4.get("score_0_100")),
             m4.get("label",""), f"ECE: {m4.get('ece_pct','N/A')}%"),
            ("M5", "Safe Refusal",         pill(m5.get("score_0_100")),
             m5.get("label",""), f"{m5.get('correct_refusals','?')}/{m5.get('total_tested','?')} correct"),
            ("M6", "Consistency",          pill(m6.get("score_0_100")),
             m6.get("label",""), f"{m6.get('paraphrases_tested','?')} paraphrases tested"),
            ("M7", "Citation Quality",     pill(m7.get("score_0_100")),
             m7.get("label",""), f"Valid: {m7.get('valid_citation_pct','?')}% | PMID: {m7.get('pmid_validity_pct','?')}%"),
        ]

        table_rows = ""
        for tag, name, score_pill, lbl, detail in rows_data:
            lbl_str = (
                f'<span style="color:var(--color-text-secondary);font-size:11px;">({lbl})</span>'
                if lbl else ""
            )
            table_rows += f"""
<tr style="border-bottom:0.5px solid var(--color-border-tertiary);">
  <td style="padding:6px 8px;font-size:12px;white-space:nowrap;">
    <strong style="color:var(--color-text-primary);">{tag}</strong>
    <span style="color:var(--color-text-secondary);"> {name}</span>
  </td>
  <td style="padding:6px 8px;text-align:right;white-space:nowrap;">{score_pill} {lbl_str}</td>
  <td style="padding:6px 8px;font-size:11px;color:var(--color-text-secondary);">{detail}</td>
</tr>"""

        # Weighted breakdown (collapsed)
        wb = sc.get("weighted_breakdown", {})
        wb_rows = ""
        for key, vals in wb.items():
            wb_rows += (
                f'<span style="font-size:11px;color:var(--color-text-secondary);">'
                f'{key.replace("_"," ").title()}: '
                f'{vals["score"]:.0f}/100 × {vals["weight"]}% = '
                f'<strong>{vals["contribution"]:.1f} pts</strong></span><br>'
            )
        weighted_section = f'''
<details style="margin-top:6px;">
  <summary style="font-size:12px;color:var(--color-text-secondary);cursor:pointer;user-select:none;">
    Weighted breakdown
  </summary>
  <div style="margin-top:4px;padding:8px;background:var(--color-background-secondary);
              border-radius:6px;line-height:1.9;">{wb_rows}</div>
</details>''' if wb_rows else ""

        return f'''
<div class="credibility-scorecard" style="margin-top:12px;padding:12px;
     border:0.5px solid var(--color-border-tertiary);border-radius:8px;">

<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
  <span style="font-weight:500;font-size:14px;color:var(--color-text-primary);">
    Credibility Scorecard
  </span>
  <span style="font-size:22px;font-weight:500;color:{total_color};">
    {total_display}/100
  </span>
  <span style="font-size:12px;font-weight:500;color:{total_color};">{label}</span>
</div>

<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <thead>
    <tr style="border-bottom:1px solid var(--color-border-tertiary);">
      <th style="text-align:left;padding:4px 8px;font-size:11px;font-weight:500;
                 color:var(--color-text-secondary);">Metric</th>
      <th style="text-align:right;padding:4px 8px;font-size:11px;font-weight:500;
                 color:var(--color-text-secondary);">Score</th>
      <th style="text-align:left;padding:4px 8px;font-size:11px;font-weight:500;
                 color:var(--color-text-secondary);">Detail</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

{weighted_section}
</div>'''

    # =========================================================================
    # OVERVIEW
    # =========================================================================

    def _render_overview(self) -> str:
        overview                = self.json_response.get("overview", "")
        overview_with_citations = self._replace_source_tags(overview)
        faithfulness_badge      = self._render_faithfulness_badge()

        return f"""**AI Overview**
{overview_with_citations}
{faithfulness_badge}
---
"""

    # =========================================================================
    # SECTION RENDERERS (all original — unchanged)
    # =========================================================================

    def _render_section_definition(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    if "name" in item:
                        name    = item.get("name", "")
                        desc    = self._replace_source_tags(item.get("description", ""))
                        details = self._replace_source_tags(item.get("details", ""))
                        lines.append(f"**{name}**: {desc}")
                        if details:
                            lines.append(f"{details}")
                        lines.append("")
                    elif "symptom" in item:
                        symptom = item.get("symptom", "")
                        desc    = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"**{symptom}**: {desc}")
                        lines.append("")
                    elif "complication" in item:
                        comp = item.get("complication", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        freq = self._clean_frequency(item.get("frequency", ""))
                        lines.append(f"**{comp}** ({freq}): {desc}" if freq else f"**{comp}**: {desc}")
                        lines.append("")
                    elif "aspect" in item:
                        aspect = item.get("aspect", "")
                        desc   = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"**{aspect}**: {desc}")
                        lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    def _render_section_complications(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    comp = item.get("complication", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    freq = self._clean_frequency(item.get("frequency", ""))
                    lines.append(f"**{comp}** ({freq}): {desc}" if freq else f"**{comp}**: {desc}")
                    lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    def _render_section_comparison(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "comparison_table" in section:
            table = section["comparison_table"]
            if table and len(table) > 0:
                lines.append("| Aspect | First Option | Second Option |")
                lines.append("|--------|-------------|---------------|")
                for row in table:
                    aspect  = row.get("aspect", "")
                    opt_a   = self._replace_source_tags(row.get("option_a", ""))
                    opt_b   = self._replace_source_tags(row.get("option_b", ""))
                    lines.append(f"| **{aspect}** | {opt_a} | {opt_b} |")
                lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    def _render_section_treatment(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    treatment = item.get("treatment", "")
                    desc      = self._replace_source_tags(item.get("description", ""))
                    lines.append(f"**{treatment}**: {desc}")
                    lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    def _render_section_diagnosis(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    procedure = item.get("procedure", "")
                    desc      = self._replace_source_tags(item.get("description", ""))
                    accuracy  = item.get("accuracy", "")
                    lines.append(f"**{procedure}**: {desc}")
                    if accuracy:
                        lines.append(f"*Accuracy: {accuracy}*")
                    lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    def _render_section_timing(self, section: Dict) -> str:
        header = section.get("header", "")
        lines  = [f"**{header}**\n"]

        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    indication  = item.get("indication", "")
                    explanation = self._replace_source_tags(item.get("explanation", ""))
                    if indication:
                        lines.append(f"**{indication}**: {explanation}")
                        lines.append("")
                    elif "consideration" in item:
                        consideration = item.get("consideration", "")
                        desc          = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"**{consideration}**: {desc}")
                        lines.append("")
        elif "content" in section:
            lines.append(self._replace_source_tags(section.get("content", "")))
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # MAIN RENDER
    # =========================================================================

    def render(self, question_type: str) -> str:
        markdown_parts = [self._render_overview()]
        sections       = self.json_response.get("sections", [])

        for section in sections:
            if question_type == "definition":
                markdown_parts.append(self._render_section_definition(section))
            elif question_type == "complications":
                markdown_parts.append(self._render_section_complications(section))
            elif question_type == "comparison":
                markdown_parts.append(self._render_section_comparison(section))
            elif question_type == "treatment":
                markdown_parts.append(self._render_section_treatment(section))
            elif question_type == "diagnosis":
                markdown_parts.append(self._render_section_diagnosis(section))
            elif question_type == "timing":
                markdown_parts.append(self._render_section_timing(section))
            else:
                markdown_parts.append(self._render_section_definition(section))

        return "\n".join(markdown_parts)

    def render_sources(self) -> str:
        """
        Render the Show Sources collapsible containing:
          1. Evidence quality
          2. Answer faithfulness  (existing)
          3. Credibility scorecard (NEW — 7 metrics)
          4. Source list
        """
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: int(x[0].replace("SOURCE_", ""))
        )

        source_lines = []
        for source_key, source in sorted_sources:
            source_num     = source_key.replace("SOURCE_", "")
            title          = source.get("title", "Unknown")
            year           = source.get("year", "")
            pmid           = source.get("pmid", "")
            evidence_level = source.get("evidence_level", "")

            source_line = f'<div id="source-{source_num}" class="source-item">'
            source_line += f"<strong>[{source_num}]</strong> {title}"
            if year and year != "Unknown":
                source_line += f" ({year})"
            details = []
            if pmid and pmid != "Unknown":
                details.append(
                    f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">PMID: {pmid}</a>'
                )
            if evidence_level and evidence_level != "Unknown":
                details.append(f"Evidence Level: {evidence_level}")
            if details:
                source_line += f" | {' | '.join(details)}"
            source_line += "</div>"
            source_lines.append(source_line)

        label     = self.confidence.get("label", "Unknown")
        score     = self.confidence.get("score", 0)
        breakdown = self.confidence.get("breakdown", "")

        faithfulness_html = self._render_faithfulness_for_sources()
        scorecard_html    = self._render_credibility_scorecard()

        return f"""
<details id="sources-section" class="sources-collapsible">
<summary class="sources-summary">📚 Show Sources</summary>

<div class="sources-content">

<div class="evidence-quality">
<strong>Evidence Quality:</strong> {label} confidence ({score}/100)
<br>
<em>Based on: {breakdown}</em>
</div>

{faithfulness_html}

{scorecard_html}

<div class="sources-divider"></div>

<div class="sources-list">
<strong>Key Sources:</strong>
<br><br>
{'<br>'.join(source_lines)}
</div>

</div>
</details>
"""

    def render_confidence(self) -> str:
        """Now included in render_sources(), returns empty for compatibility."""
        return ""
