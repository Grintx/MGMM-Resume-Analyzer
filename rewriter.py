# rewriter.py
"""
Refined ResumeRewriter

Key features:
- Extract text from PDF (PyPDF2 fallback).
- Hybrid section extraction (spaCy Matcher preferred, regex fallback).
- Safe regex usage (flags passed to re.compile).
- Bullet & whitespace normalization.
- AI rewriting via Flan-T5 (graceful fallback to rule-based rewrites).
- PDF generation via ReportLab (primary). If weasyprint is installed, can optionally use it.
- TXT export for single-block optimized resume.
- Public API returns dict compatible with app.py: {"text","sections","improvements"}.
"""

import re
import json
import os
from typing import Dict, List, Any, Optional
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transformers (Flan-T5) for optional AI rewriting
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    T5ForConditionalGeneration = None
    T5Tokenizer = None
    torch = None
    _TRANSFORMERS_AVAILABLE = False

# Optional spaCy for better section detection
try:
    import spacy
    from spacy.matcher import Matcher
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    Matcher = None
    _SPACY_AVAILABLE = False

# Primary PDF engine: ReportLab (keeps your current code compatible)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

# Optional WeasyPrint for HTML/CSS-based PDFs
try:
    from weasyprint import HTML
    _WEASYPRINT_AVAILABLE = True
except Exception:
    _WEASYPRINT_AVAILABLE = False

# Try PyPDF2 for PDF text extraction
try:
    from PyPDF2 import PdfReader
    _PYPDF2_AVAILABLE = True
except Exception:
    PdfReader = None
    _PYPDF2_AVAILABLE = False


class ResumeRewriter:
    """
    ResumeRewriter: produces an optimized resume text and can export PDF/TXT.
    Public methods:
      - rewrite_resume(resume_data, job_description, ats_scores, sections=None, style='professional')
      - generate_pdf(optimized_resume, template_style='professional')
      - save_txt(optimized_resume, filename)
      - extract_text_from_pdf(path)
    """

    def __init__(self, t5_model_name: str = "google/flan-t5-base"):
        # Optional AI model
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
                # Use CPU if CUDA not available
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model_loaded = True
                logger.info("Flan-T5 model loaded for rewriting.")
            except Exception as e:
                logger.warning(f"Could not load Flan-T5 model: {e}")
                self.model_loaded = False
        else:
            logger.info("Transformers not available; using fallback rewriting.")

        # spaCy: load lightweight model if available
        self.spacy_loaded = False
        self.nlp = None
        self.matcher = None
        if _SPACY_AVAILABLE:
            try:
                # Prefer a small installed model, otherwise create a blank pipeline
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # If user installed other model names, attempt load; otherwise blank
                    try:
                        self.nlp = spacy.load("en_core_web_md")
                    except Exception:
                        self.nlp = spacy.blank("en")
                self.matcher = Matcher(self.nlp.vocab)
                self._init_spacy_matcher()
                self.spacy_loaded = True
                logger.info("spaCy loaded for section extraction.")
            except Exception as e:
                logger.warning(f"spaCy model not loaded; falling back to regex extraction: {e}")
                self.spacy_loaded = False
        else:
            logger.info("spaCy not installed; will use regex fallback for section extraction.")

        # Rewriting prompt templates
        self.rewriting_templates = {
            "experience": (
                "Rewrite these experience bullets to be concise, ATS-friendly, and action-oriented. "
                "Use strong action verbs, quantify achievements where possible, remove personal pronouns. "
                "Make each bullet 1-2 lines.\n\nBullets:\n{text}\n\nJob context:\n{context}\n\nRewritten:"
            ),
            "skills": (
                "Optimize skills section: group similar skills, use canonical names, order by relevance.\n\n"
                "Skills:\n{text}\n\nJob context:\n{context}\n\nOptimized:"
            ),
            "summary": (
                "Create a professional, ATS-optimized summary (3-4 sentences). "
                "Start with years of experience where present and include top keywords.\n\n"
                "Background:\n{text}\n\nJob:\n{context}\n\nSummary:"
            ),
            "education": (
                "Format the education entries for ATS: include degree, institution, graduation year and relevant coursework if present.\n\n"
                "Education:\n{text}\n\nJob:\n{context}\n\nOptimized:"
            )
        }

        # Resume template style options (used for PDF layout)
        self.resume_templates = {
            "professional": {
                "font_name": "Helvetica",
                "font_size": 10,
                "section_spacing": 0.18,
                "header_size": 16
            },
            "modern": {
                "font_name": "Helvetica",
                "font_size": 10,
                "section_spacing": 0.14,
                "header_size": 15
            },
            "creative": {
                "font_name": "Helvetica-Bold",
                "font_size": 10,
                "section_spacing": 0.20,
                "header_size": 16
            },
            "minimal": {
                "font_name": "Times-Roman",
                "font_size": 10,
                "section_spacing": 0.22,
                "header_size": 14
            }
        }

    # -----------------------
    # Public API
    # -----------------------
    def rewrite_resume(
        self,
        resume_data: Dict[str, Any],
        job_description: str,
        ats_scores: Dict[str, Any],
        sections: Optional[List[str]] = None,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Main entry point.
        - resume_data: parsed resume dict (must contain at least 'text')
        - job_description: JD text
        - ats_scores: scores dict (optional, used for recommendations)
        - sections: list of section names to optimize (strings like "Experience","Skills")
        - style: template style for PDF generation
        Returns: dict with keys: 'text' (single block), 'sections' (per-section optimized), 'improvements' (list)
        """
        raw_text = resume_data.get("text", "") or ""
        if not raw_text.strip():
            return {"text": "", "sections": {}, "improvements": ["No resume text provided"]}

        # normalize requested sections
        if sections is None:
            target_sections = ["experience", "skills", "summary"]
        else:
            target_sections = [s.lower() for s in sections]

        # Extract sections (dict of raw content)
        extracted = self._extract_sections(raw_text)

        optimized_sections: Dict[str, str] = {}
        improvements: List[str] = []

        # For each requested section, rewrite (AI if available else fallback)
        for sec in target_sections:
            raw = extracted.get(sec, "").strip()
            if not raw:
                logger.debug(f"Section '{sec}' not found in extracted text.")
                continue

            rewritten = ""
            # try model
            if self.model_loaded:
                rewritten = self._rewrite_with_model(raw, job_description, sec)
                if rewritten:
                    improvements.append(f"Rewrote {sec} with Flan-T5")
            # fallback
            if not rewritten:
                rewritten = self._fallback_rewrite(raw, sec)
                improvements.append(f"Applied rule-based improvements to {sec}")

            # final cleanup/normalization
            rewritten = self._cleanup_section_text(rewritten)
            optimized_sections[sec] = rewritten

        # keep other extracted sections (cleaned) so final resume is complete
        for k, v in extracted.items():
            if k not in optimized_sections:
                optimized_sections[k] = self._cleanup_section_text(v)

        # Compose single block text with formatted headers (one-block TXT requested)
        composed_text = self._compose_single_block(optimized_sections, resume_data)

        # Ensure a reasonable summary exists; auto-generate if missing or too short
        if ("summary" not in optimized_sections) or (len(optimized_sections.get("summary", "")) < 30):
            summary_auto = self._generate_professional_summary(resume_data, optimized_sections, job_description)
            if summary_auto:
                optimized_sections["summary"] = summary_auto
                improvements.append("Auto-generated ATS-optimized professional summary")
                composed_text = self._compose_single_block(optimized_sections, resume_data)

        result = {
            "text": composed_text,
            "sections": optimized_sections,
            "improvements": improvements
        }
        return result

    # -----------------------
    # PDF text extraction helper
    # -----------------------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2 if available.
        Returns the text (empty string on failure).
        """
        if not os.path.exists(pdf_path):
            logger.error("PDF path does not exist: %s", pdf_path)
            return ""

        if _PYPDF2_AVAILABLE and PdfReader is not None:
            try:
                text_parts = []
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    try:
                        txt = page.extract_text()
                        if txt:
                            text_parts.append(txt)
                    except Exception:
                        continue
                return "\n\n".join(text_parts).strip()
            except Exception as e:
                logger.warning("PyPDF2 extraction failed: %s", e)

        # Fallback: try reading as binary and guessing (limited)
        try:
            with open(pdf_path, "rb") as f:
                data = f.read()
            # as a last resort attempt naive decode
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        except Exception as e:
            logger.error("Unable to open PDF for extraction: %s", e)
            return ""

    # -----------------------
    # Section extraction (spaCy + regex fallback)
    # -----------------------
    def _init_spacy_matcher(self):
        """Initialize spaCy Matcher with some heading patterns"""
        if not self.nlp or not self.matcher:
            return
        try:
            # clear any existing rules
            for key in list(self.matcher):
                try:
                    self.matcher.remove(key)
                except Exception:
                    pass
        except Exception:
            pass

        defs = {
            "experience": [["experience"], ["work", "experience"], ["professional", "experience"], ["employment"]],
            "skills": [["skills"], ["technical", "skills"], ["competencies"], ["proficiencies"]],
            "summary": [["summary"], ["professional", "summary"], ["profile"], ["objective"], ["about"]],
            "education": [["education"], ["academic", "background"], ["qualifications"], ["certifications"]],
            "projects": [["projects"], ["personal", "projects"]],
        }

        for label, phraselist in defs.items():
            patterns = []
            for phrase in phraselist:
                patterns.append([{"LOWER": tok} for tok in phrase])
            try:
                self.matcher.add(label.upper(), patterns)
            except Exception:
                # ignore duplicates or unsupported ops
                pass

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Hybrid extraction:
        - If spaCy loaded: find headings using matcher and slice between headings.
        - Otherwise: fallback to regex headings search (safe compile with flags).
        Returns a dict: section_name -> raw section text (no header).
        """
        sections: Dict[str, str] = {}

        if self.spacy_loaded and self.nlp and self.matcher:
            try:
                doc = self.nlp(text)
                matches = self.matcher(doc)
                found = []
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id].lower()
                    span = doc[start:end]
                    found.append((span.start_char, span.end_char, label, span.text))
                if not found:
                    return self._regex_extract_sections(text)
                found = sorted(found, key=lambda x: x[0])

                for idx, (spos, epos, label, span_text) in enumerate(found):
                    start_idx = epos
                    end_idx = found[idx + 1][0] if idx + 1 < len(found) else len(text)
                    raw = text[start_idx:end_idx].strip()
                    if raw:
                        sections[label] = raw
                sections.setdefault("full", text.strip())
                return sections
            except Exception as e:
                logger.warning("spaCy extraction failed: %s", e)
                return self._regex_extract_sections(text)
        else:
            return self._regex_extract_sections(text)

    def _regex_extract_sections(self, text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        patterns = {
            "experience": r"(?:^|\n)\s*(?:work\s+experience|professional\s+experience|experience|employment)\s*(?:\n|:)",
            "skills": r"(?:^|\n)\s*(?:technical\s+skills|skills|competencies|proficiencies)\s*(?:\n|:)",
            "summary": r"(?:^|\n)\s*(?:professional\s+summary|summary|profile|objective|about)\s*(?:\n|:)",
            "education": r"(?:^|\n)\s*(?:education|academic\s+background|qualifications|certifications)\s*(?:\n|:)"
        }

        headings = []
        for key, pat in patterns.items():
            try:
                regex = re.compile(pat, flags=re.IGNORECASE)
                for m in regex.finditer(text):
                    headings.append((m.start(), m.end(), key))
            except re.error:
                continue

        if not headings:
            out = {"full": text.strip()}
            # heuristics for summary and skills
            first_para = text.strip().split("\n\n")[0]
            if len(first_para.split()) < 80:
                out["summary"] = first_para.strip()
            skills_search = re.search(r"(?:Skills|Technical Skills)[:\s]*(.+)", text, flags=re.IGNORECASE)
            if skills_search:
                out["skills"] = skills_search.group(1).strip()
            return out

        headings = sorted(headings, key=lambda x: x[0])
        for i, (sstart, send, key) in enumerate(headings):
            content_start = send
            content_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            snippet = text[content_start:content_end].strip()
            if snippet:
                sections[key] = snippet

        sections.setdefault("full", text.strip())
        return sections

    # -----------------------
    # Rewriting (AI + fallback)
    # -----------------------
    def _rewrite_with_model(self, content: str, job_description: str, section_type: str) -> str:
        """Use Flan-T5 to rewrite. Returns rewritten text or empty string on failure."""
        if not self.model_loaded or not self.model or not self.tokenizer:
            return ""

        template = self.rewriting_templates.get(section_type, "{text}\n\n{context}")
        prompt = template.format(text=self._prepare_text_for_model(content, section_type), context=job_description[:500])

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=256,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:
            logger.warning("AI model rewrite failed for %s: %s", section_type, e)
            return ""

    def _prepare_text_for_model(self, content: str, section_type: str) -> str:
        cleaned = self._cleanup_section_text(content, preserve_bullets=True)
        return cleaned[:1200]

    def _fallback_rewrite(self, content: str, section_type: str) -> str:
        text = content.strip()
        if section_type == "experience":
            return self._rule_improve_experience(text)
        if section_type == "skills":
            return self._rule_improve_skills(text)
        if section_type == "summary":
            return self._rule_improve_summary(text)
        if section_type == "education":
            return self._rule_improve_education(text)
        return text

    # -----------------------
    # Rule-based improvements
    # -----------------------
    def _rule_improve_experience(self, content: str) -> str:
        lines = re.split(r'\n+', content)
        improved = []
        weak_map = {
            r'\bwas responsible for\b': 'managed',
            r'\bworked on\b': 'developed',
            r'\bhelped with\b': 'supported',
            r'\bparticipated in\b': 'contributed to'
        }
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r'^[\-\u2022\*\d\.\)]+\s*', '', ln)
            for patt, rep in weak_map.items():
                ln = re.sub(patt, rep, ln, flags=re.IGNORECASE)
            ln = re.sub(r'\bI\b', '', ln, flags=re.IGNORECASE)
            ln = re.sub(r'\bmy\b', 'the', ln, flags=re.IGNORECASE)
            if not ln.startswith('•'):
                ln = '• ' + ln
            words = ln.split()
            if len(words) > 45:
                ln = ' '.join(words[:40]) + '...'
            improved.append(ln)
        return '\n'.join(improved)

    def _rule_improve_skills(self, content: str) -> str:
        items = re.split(r'[,\n;/•\|\-]+', content)
        normalized = []
        seen = set()
        for it in items:
            s = it.strip()
            if not s:
                continue
            s_norm = s.title()
            if s_norm not in seen:
                normalized.append(s_norm)
                seen.add(s_norm)
        languages = []
        frameworks = []
        tools = []
        other = []
        lang_keys = ['Python', 'Java', 'Javascript', 'C++', 'C#', 'Go', 'Rust', 'Kotlin', 'Swift', 'Php', 'Ruby']
        framework_keys = ['React', 'Angular', 'Vue', 'Django', 'Flask', 'Spring', 'Express', 'Node']
        tool_keys = ['Docker', 'Kubernetes', 'Aws', 'Azure', 'Gcp', 'Git', 'Jenkins', 'Sql', 'Postgres', 'Mysql', 'Mongo']
        for s in normalized:
            sl = s.lower()
            if any(k.lower() in sl for k in lang_keys):
                languages.append(s)
            elif any(k.lower() in sl for k in framework_keys):
                frameworks.append(s)
            elif any(k.lower() in sl for k in tool_keys):
                tools.append(s)
            else:
                other.append(s)
        out = []
        if languages:
            out.append("Programming Languages: " + ", ".join(languages))
        if frameworks:
            out.append("Frameworks & Libraries: " + ", ".join(frameworks))
        if tools:
            out.append("Tools & Technologies: " + ", ".join(tools))
        if other:
            out.append("Other Skills: " + ", ".join(other))
        return "\n".join(out)

    def _rule_improve_summary(self, content: str) -> str:
        text = re.sub(r'\s+', ' ', content).strip()
        years = ""
        m = re.search(r'(\d+)\s*\+?\s*(?:years?|yrs?)', text, flags=re.IGNORECASE)
        if m:
            years = f"{m.group(1)}+ years experience"
        skills_snippet = ""
        skills_match = re.search(r'(?:skills|technical skills)[:\s]*([^\n]{0,120})', content, flags=re.IGNORECASE)
        if skills_match:
            skills_snippet = skills_match.group(1).strip()
        base = text.split('\n')[0][:250]
        parts = []
        if years:
            parts.append(f"Experienced professional with {years}.")
        if base:
            parts.append(base.strip().rstrip('.'))
        if skills_snippet:
            parts.append(f"Skilled in {skills_snippet.strip().rstrip('.')}.")
        summary = " ".join(parts).strip()
        if not summary.endswith('.'):
            summary += '.'
        words = summary.split()
        if len(words) > 80:
            summary = " ".join(words[:75]) + "."
        return summary

    def _rule_improve_education(self, content: str) -> str:
        lines = re.split(r'\n+', content)
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r'\bB\.?S\.?\b', 'Bachelor of Science', ln, flags=re.IGNORECASE)
            ln = re.sub(r'\bM\.?S\.?\b', 'Master of Science', ln, flags=re.IGNORECASE)
            ln = re.sub(r'\bB\.?A\.?\b', 'Bachelor of Arts', ln, flags=re.IGNORECASE)
            out.append(ln)
        return "\n".join(out)

    # -----------------------
    # Cleanup & composition helpers
    # -----------------------
    def _cleanup_section_text(self, content: str, preserve_bullets: bool = False) -> str:
        if not content:
            return ""
        text = content.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[•\u2022\*\-–—]+', '•', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'(^|\n)\s*•\s*', r'\1• ', text)
        lines = [ln.rstrip() for ln in text.split('\n')]
        if not preserve_bullets:
            normalized_lines = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    normalized_lines.append("")
                    continue
                if not ln.startswith('•') and len(ln) < 140 and len(ln.split()) <= 22:
                    normalized_lines.append('• ' + ln)
                else:
                    normalized_lines.append(ln)
            text = "\n".join(normalized_lines)
        else:
            text = "\n".join(lines)
        return text.strip()

    def _compose_single_block(self, sections: Dict[str, str], resume_data: Dict[str, Any]) -> str:
        order = ["summary", "experience", "skills", "education", "projects", "full"]
        parts: List[str] = []
        contact = resume_data.get("contact_info", {})
        contact_lines = []
        if contact.get("email"):
            contact_lines.append(f"Email: {contact.get('email')}")
        if contact.get("phone"):
            contact_lines.append(f"Phone: {contact.get('phone')}")
        if contact.get("linkedin"):
            contact_lines.append(f"LinkedIn: {contact.get('linkedin')}")
        if contact.get("location"):
            contact_lines.append(f"Location: {contact.get('location')}")
        if contact_lines:
            parts.append("CONTACT INFORMATION")
            parts.append("\n".join(contact_lines))
            parts.append("")
        for key in order:
            if key in sections and sections[key].strip():
                header = key.upper()
                parts.append(header)
                parts.append(sections[key].strip())
                parts.append("")
        if "full" in sections and not parts:
            parts.append("RESUME")
            parts.append(sections["full"])
        single = "\n\n".join([p.strip() for p in parts if p is not None and p != ""])
        single = re.sub(r'\n{3,}', '\n\n', single).strip()
        return single

    def _generate_professional_summary(self, resume_data: Dict[str, Any], optimized_sections: Dict[str, str], job_description: str) -> str:
        full_text = resume_data.get("text", "")
        years_m = re.search(r'(\d+)\s*\+?\s*(?:years?|yrs?)', full_text, flags=re.IGNORECASE)
        years_str = f"{years_m.group(1)}+ years" if years_m else ""
        skills_block = optimized_sections.get("skills", "") or ", ".join(resume_data.get("skills", [])[:6])
        skill_names = []
        if skills_block:
            tokens = re.split(r'[,:;\n]+', skills_block)
            for t in tokens:
                t = t.strip()
                if t:
                    t = re.sub(r'^[A-Za-z\s]+:\s*', '', t)
                    skill_names.append(t.split()[0])
                if len(skill_names) >= 6:
                    break
        skills_str = ", ".join(skill_names[:6])
        role_match = re.search(r'\b(senior|lead|manager|engineer|analyst|developer)\b', job_description, flags=re.IGNORECASE)
        role_hint = role_match.group(1).capitalize() if role_match else "Professional"
        pieces = []
        if years_str:
            pieces.append(f"{role_hint} with {years_str} of experience")
        else:
            pieces.append(f"{role_hint} with experience in relevant domains")
        if skills_str:
            pieces.append(f"Skilled in {skills_str}")
        pieces.append("Proven ability to deliver measurable results and collaborate across teams.")
        summary = ". ".join(pieces).strip()
        if not summary.endswith("."):
            summary += "."
        words = summary.split()
        if len(words) > 75:
            summary = " ".join(words[:70]) + "."
        return summary

    # -----------------------
    # PDF & TXT generation
    # -----------------------
    def generate_pdf(self, optimized_resume: Dict[str, Any], template_style: str = "professional") -> str:
        text = optimized_resume.get("text", "") or ""
        if not text:
            return ""
        out_filename = f"optimized_resume_{template_style}.pdf"
        style_cfg = self.resume_templates.get(template_style, self.resume_templates["professional"])
        # Try WeasyPrint first for nicer styling
        if _WEASYPRINT_AVAILABLE:
            try:
                html = self._to_html_for_pdf(text, style_cfg)
                HTML(string=html).write_pdf(out_filename)
                return out_filename
            except Exception as e:
                logger.warning("WeasyPrint generation failed, falling back to ReportLab: %s", e)
        if not _REPORTLAB_AVAILABLE:
            logger.error("No PDF engine available (weasyprint/reportlab).")
            return ""
        try:
            styles = getSampleStyleSheet()
            doc = SimpleDocTemplate(out_filename, pagesize=letter,
                                    topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                                    leftMargin=0.6 * inch, rightMargin=0.6 * inch)
            header_style = ParagraphStyle(
                "Header", parent=styles["Heading1"],
                fontName=style_cfg["font_name"], fontSize=style_cfg["header_size"], leading=style_cfg["header_size"] + 2,
                spaceAfter=6
            )
            section_style = ParagraphStyle(
                "SectionHeader", parent=styles["Heading2"],
                fontName=style_cfg["font_name"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=6
            )
            body_style = ParagraphStyle(
                "Body", parent=styles["Normal"],
                fontName=style_cfg["font_name"], fontSize=style_cfg["font_size"], leading=12
            )
            bullet_style = ParagraphStyle(
                "Bullet", parent=body_style,
                leftIndent=18, bulletIndent=8
            )
            story = []
            parts = re.split(r'\n{2,}', text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                lines = part.split("\n")
                first = lines[0].strip()
                if first.isupper() and len(first) <= 40:
                    story.append(Paragraph(first, section_style))
                    content_lines = "\n".join(lines[1:]).strip()
                    for ln in content_lines.split("\n"):
                        ln = ln.strip()
                        if not ln:
                            continue
                        if ln.startswith("•"):
                            text_line = ln[1:].strip()
                            story.append(Paragraph(f"• {self._escape_para(text_line)}", bullet_style))
                        else:
                            story.append(Paragraph(self._escape_para(ln), body_style))
                else:
                    for ln in part.split("\n"):
                        ln = ln.strip()
                        if not ln:
                            continue
                        if ln.startswith("•"):
                            story.append(Paragraph(f"• {self._escape_para(ln[1:].strip())}", bullet_style))
                        else:
                            story.append(Paragraph(self._escape_para(ln), body_style))
                story.append(Spacer(1, style_cfg["section_spacing"] * inch))
            doc.build(story)
            return out_filename
        except Exception as e:
            logger.exception("ReportLab PDF generation failed: %s", e)
            return ""

    def save_txt(self, optimized_resume: Dict[str, Any], filename: str = "optimized_resume.txt") -> str:
        """
        Save the single-block optimized resume to a TXT file and return its path.
        """
        text = optimized_resume.get("text", "") or ""
        if not text:
            return ""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            return filename
        except Exception as e:
            logger.exception("Saving TXT failed: %s", e)
            return ""

    def _to_html_for_pdf(self, text: str, style_cfg: Dict[str, Any]) -> str:
        html_parts = []
        for block in re.split(r'\n{2,}', text):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            head = lines[0].strip()
            if head.isupper() and len(head) <= 40:
                html_parts.append(f"<h3 class='section'>{self._escape_html(head.title())}</h3>")
                html_body = "\n".join(lines[1:])
                bullets = []
                paras = []
                for ln in html_body.split("\n"):
                    ln = ln.strip()
                    if ln.startswith("•"):
                        bullets.append(f"<li>{self._escape_html(ln[1:].strip())}</li>")
                    elif ln:
                        paras.append(f"<p>{self._escape_html(ln)}</p>")
                if bullets:
                    html_parts.append(f"<ul>{''.join(bullets)}</ul>")
                if paras:
                    html_parts.extend(paras)
            else:
                html_parts.append("<p>" + self._escape_html(block).replace("\n", "<br/>") + "</p>")
        html_body = "\n".join(html_parts)
        css = """
        body { font-family: Arial, Helvetica, sans-serif; color: #111; margin: 28px; }
        h3.section { font-size: 13px; color: #222; margin-top: 12px; text-transform: uppercase; }
        ul { margin: 6px 0 12px 18px; }
        p { margin: 4px 0; font-size: 11px; }
        """
        html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
        return html

    def generate_latex(self, optimized_resume: Dict[str, Any]) -> str:
        text = optimized_resume.get("text", "")
        lines = []
        lines.append(r"\documentclass[11pt]{article}")
        lines.append(r"\usepackage[margin=0.7in]{geometry}")
        lines.append(r"\usepackage{enumitem}")
        lines.append(r"\begin{document}")
        for block in re.split(r'\n{2,}', text):
            block = block.strip()
            if not block:
                continue
            parts = block.split("\n")
            header = parts[0].strip()
            if header.isupper():
                lines.append(r"\section*{" + self._latex_escape(header.title()) + "}")
                content_lines = parts[1:]
                bullets = [ln for ln in content_lines if ln.strip().startswith("•")]
                pars = [ln for ln in content_lines if not ln.strip().startswith("•") and ln.strip()]
                for p in pars:
                    lines.append(self._latex_escape(p.strip()) + "\n")
                if bullets:
                    lines.append(r"\begin{itemize}[leftmargin=*]")
                    for b in bullets:
                        lines.append(r"\item " + self._latex_escape(b.lstrip("•").strip()))
                    lines.append(r"\end{itemize}")
            else:
                lines.append(self._latex_escape(block))
        lines.append(r"\end{document}")
        return "\n".join(lines)

    # -----------------------
    # Small escaping helpers
    # -----------------------
    def _escape_html(self, s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))

    def _escape_para(self, s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _latex_escape(self, s: str) -> str:
        replacements = {
            "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\^{}", "\\": r"\textbackslash{}"
        }
        res = s
        for k, v in replacements.items():
            res = res.replace(k, v)
        return res
