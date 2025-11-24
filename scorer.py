"""
scorer_fixed.py

Complete ATSScorer class with fixes for JSON serialization of numpy types
and explicit float conversions to avoid `float32` JSON errors.

Changes:
- ADDED: to_serializable(obj) utility to convert numpy types to native types
- UPDATED: explicit float(...) conversions for all numeric score outputs
- UPDATED: completed _get_compliance_grade and ensured full class closing
"""

import re
import json
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk

# Download required NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ADDED: utility to convert numpy types (and nested structures) to Python-native types
def to_serializable(obj):
    """
    Convert numpy types and arrays to JSON-serializable Python native types.
    Recurses into lists/dicts.

    Examples:
      np.float32(0.5) -> 0.5 (float)
      np.array([1,2]) -> [1,2] (list)
      {'a': np.int64(2)} -> {'a': 2}
    """
    # Primitive python types -> return as-is
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # numpy scalar
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # lists / tuples -> convert each element
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]

    # dict -> convert values
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    # objects with .item() (numpy scalar-like)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    # fallback: attempt json.dumps; if fails, return string repr
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class ATSScorer:
    """Advanced ATS scoring system with multiple similarity metrics"""

    def __init__(self):
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )

        # Skill categories and weightings
        self.skill_categories = {
            'technical_skills': {
                'weight': 0.3,
                'keywords': [
                    'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
                    'machine learning', 'data analysis', 'cloud computing', 'aws', 'docker',
                    'kubernetes', 'git', 'agile', 'scrum', 'rest api', 'microservices'
                ]
            },
            'soft_skills': {
                'weight': 0.2,
                'keywords': [
                    'leadership', 'communication', 'teamwork', 'problem solving',
                    'project management', 'analytical', 'creative thinking', 'adaptability',
                    'time management', 'critical thinking', 'collaboration', 'innovation'
                ]
            },
            'industry_keywords': {
                'weight': 0.25,
                'keywords': [
                    'experience', 'development', 'management', 'analysis', 'design',
                    'implementation', 'optimization', 'strategy', 'research', 'support'
                ]
            },
            'achievements': {
                'weight': 0.25,
                'keywords': [
                    'improved', 'increased', 'reduced', 'achieved', 'delivered',
                    'managed', 'led', 'developed', 'created', 'implemented'
                ]
            }
        }

        # ATS scoring weights (configurable)
        self.scoring_weights = {
            'sbert_similarity': 0.5,
            'tfidf_similarity': 0.3,
            'layout_compliance': 0.2
        }

    def compute_ats_score(self, resume_data: Dict, job_description: str,
                         analysis_results: Dict = None) -> Dict[str, Any]:
        """Compute comprehensive ATS score"""

        resume_text = resume_data.get('text', '')

        # Compute individual similarity scores (ensure native floats)
        sbert_score = float(self._compute_sbert_similarity(resume_text, job_description))
        tfidf_score = float(self._compute_tfidf_similarity(resume_text, job_description))
        layout_score = float(analysis_results.get('layout_score', 0.5)) if analysis_results else 0.5

        # Skill gap analysis
        skill_analysis = self._analyze_skill_gaps(resume_text, job_description)

        # Keyword matching analysis
        keyword_analysis = self._analyze_keyword_matching(resume_text, job_description)

        # Calculate weighted final score (explicit float)
        final_score = float(
            self.scoring_weights['sbert_similarity'] * sbert_score +
            self.scoring_weights['tfidf_similarity'] * tfidf_score +
            self.scoring_weights['layout_compliance'] * layout_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sbert_score, tfidf_score, layout_score, skill_analysis, keyword_analysis
        )

        # Build result dict and convert numpy types to python natives before returning
        result = {
            'final_score': final_score,
            'sbert_score': sbert_score,
            'tfidf_score': tfidf_score,
            'layout_score': layout_score,
            'skill_analysis': skill_analysis,
            'keyword_analysis': keyword_analysis,
            'missing_skills': skill_analysis.get('missing_skills', []),
            'matching_skills': skill_analysis.get('matching_skills', []),
            'recommendations': recommendations,
            'score_breakdown': {
                'semantic_similarity': f"{sbert_score:.1%}",
                'keyword_match': f"{tfidf_score:.1%}",
                'ats_formatting': f"{layout_score:.1%}"
            }
        }

        # ADDED: make serializable (convert numpy types to native python types)
        return to_serializable(result)

    def _compute_sbert_similarity(self, resume_text: str, job_description: str) -> float:
        """Compute semantic similarity using SBERT embeddings"""
        if not self.sentence_model:
            return 0.5  # Fallback score

        try:
            # Clean and preprocess texts
            resume_clean = self._preprocess_text(resume_text)
            jd_clean = self._preprocess_text(job_description)

            # Generate embeddings (returns numpy arrays)
            resume_embedding = self.sentence_model.encode([resume_clean])
            jd_embedding = self.sentence_model.encode([jd_clean])

            # Compute cosine similarity
            similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]

            # Ensure numeric python float is returned
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            print(f"SBERT similarity computation failed: {e}")
            return 0.5

    def _compute_tfidf_similarity(self, resume_text: str, job_description: str) -> float:
        """Compute TF-IDF based keyword similarity"""
        try:
            # Preprocess texts
            resume_clean = self._preprocess_text(resume_text)
            jd_clean = self._preprocess_text(job_description)

            # Fit TF-IDF on both documents
            documents = [resume_clean, jd_clean]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            print(f"TF-IDF similarity computation failed: {e}")
            return 0.5

    def _analyze_skill_gaps(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze skill gaps between resume and job description"""

        # Extract skills from both texts
        resume_skills = self._extract_skills_from_text(resume_text.lower())
        jd_skills = self._extract_skills_from_text(job_description.lower())

        # Find matching and missing skills
        matching_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        extra_skills = list(set(resume_skills) - set(jd_skills))

        # Calculate skill match percentage
        total_jd_skills = len(jd_skills)
        skill_match_percentage = float(len(matching_skills) / total_jd_skills) if total_jd_skills > 0 else 0.0

        # Categorize missing skills by importance
        critical_missing = []
        important_missing = []
        nice_to_have_missing = []

        for skill in missing_skills:
            skill_frequency = job_description.lower().count(skill)
            if skill_frequency >= 3:
                critical_missing.append(skill)
            elif skill_frequency >= 2:
                important_missing.append(skill)
            else:
                nice_to_have_missing.append(skill)

        return {
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'extra_skills': extra_skills,
            'skill_match_percentage': skill_match_percentage,
            'critical_missing': critical_missing,
            'important_missing': important_missing,
            'nice_to_have_missing': nice_to_have_missing,
            'total_resume_skills': len(resume_skills),
            'total_jd_skills': total_jd_skills
        }

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching and keyword lists"""
        skills = set()

        # Use predefined skill keywords
        for category in self.skill_categories.values():
            for skill in category['keywords']:
                if skill in text:
                    skills.add(skill)

        # Additional technical skills extraction
        tech_patterns = [
            r'\b(?:python|java|javascript|c\+\+|c#|php|ruby|go|rust|swift|kotlin)\b',
            r'\b(?:react|angular|vue|django|flask|spring|express|nodejs)\b',
            r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|oracle)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git|linux)\b',
            r'\b(?:machine learning|artificial intelligence|data science|deep learning)\b',
            r'\b(?:rest api|microservices|devops|ci/cd|agile|scrum)\b'
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([match.lower() for match in matches])

        # Extract skills from skills section if present
        skills_section_match = re.search(
            r'(?:skills|technical skills|core competencies)[:\s]*(.*?)(?=\n\s*[A-Z]|\n\s*\n|$)',
            text, re.IGNORECASE | re.DOTALL
        )

        if skills_section_match:
            skills_text = skills_section_match.group(1)
            # Split by common delimiters
            skill_items = re.split(r'[,•·\n\-\|/]', skills_text)
            for skill in skill_items:
                skill = skill.strip().lower()
                if skill and len(skill) > 1 and len(skill) < 30:
                    skills.add(skill)

        return list(skills)

    def _analyze_keyword_matching(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze keyword matching between resume and job description"""

        # Extract important keywords from job description
        jd_keywords = self._extract_important_keywords(job_description)
        resume_keywords = self._extract_important_keywords(resume_text)

        # Find matches
        matched_keywords = []
        missing_keywords = []

        for keyword in jd_keywords:
            if keyword.lower() in resume_text.lower():
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        # Calculate keyword density
        keyword_density = float(len(matched_keywords) / len(jd_keywords)) if jd_keywords else 0.0

        # Analyze keyword frequency
        keyword_frequency = {}
        for keyword in matched_keywords:
            frequency = resume_text.lower().count(keyword.lower())
            keyword_frequency[keyword] = int(frequency)

        return {
            'jd_keywords': jd_keywords,
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
            'keyword_density': keyword_density,
            'keyword_frequency': keyword_frequency,
            'total_jd_keywords': len(jd_keywords),
            'total_matched': len(matched_keywords)
        }

    def _extract_important_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TF-IDF and pattern matching"""
        keywords = set()

        # Use TF-IDF to find important terms
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )

            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top keywords by TF-IDF score
            top_indices = np.argsort(tfidf_scores)[-20:]  # Top 20 keywords
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    keywords.add(feature_names[idx])

        except Exception as e:
            print(f"TF-IDF keyword extraction failed: {e}")

        # Add domain-specific keywords
        domain_patterns = [
            r'\b(?:required|must have|essential|mandatory)[:\s]*(.*?)(?=\n|\.)',
            r'\b(?:experience with|proficiency in|knowledge of)[:\s]*(.*?)(?=\n|\.)',
            r'\b(?:bachelor|master|phd|degree)[:\s]*(.*?)(?=\n|\.)',
            r'\b(?:\d+\+?\s*years?)[:\s]*(.*?)(?=\n|\.)'
        ]

        for pattern in domain_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and split the match
                terms = re.split(r'[,;]', match)
                for term in terms:
                    term = term.strip()
                    if term and len(term) > 2 and len(term) < 50:
                        keywords.add(term.lower())

        return list(keywords)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity computation"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\-]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove very short words
        words = text.split()
        words = [word for word in words if len(word) > 2]

        return ' '.join(words)

    def _generate_recommendations(self, sbert_score: float, tfidf_score: float,
                                  layout_score: float, skill_analysis: Dict,
                                  keyword_analysis: Dict) -> List[str]:
        """Generate personalized recommendations based on scores"""
        recommendations = []

        # SBERT score recommendations
        if sbert_score < 0.6:
            recommendations.append(
                "Improve semantic alignment by using similar language and concepts from the job description"
            )
            recommendations.append(
                "Consider reorganizing content to better match the job requirements structure"
            )

        # TF-IDF score recommendations
        if tfidf_score < 0.5:
            recommendations.append(
                "Include more relevant keywords from the job description in your resume"
            )
            recommendations.append(
                "Match the terminology used in the job posting more closely"
            )

        # Layout recommendations
        if layout_score < 0.7:
            recommendations.append(
                "Simplify formatting to improve ATS compatibility"
            )
            recommendations.append(
                "Use standard section headers like 'Experience', 'Education', 'Skills'"
            )
            recommendations.append(
                "Avoid complex layouts, tables, or multi-column formats"
            )

        # Skill-based recommendations
        critical_missing = skill_analysis.get('critical_missing', [])
        if critical_missing:
            recommendations.append(
                f"Consider highlighting experience with: {', '.join(critical_missing[:5])}"
            )

        important_missing = skill_analysis.get('important_missing', [])
        if important_missing:
            recommendations.append(
                f"Add relevant experience or training in: {', '.join(important_missing[:3])}"
            )

        # Keyword density recommendations
        if keyword_analysis.get('keyword_density', 0) < 0.3:
            missing_keywords = keyword_analysis.get('missing_keywords', [])[:5]
            if missing_keywords:
                recommendations.append(
                    f"Include these important keywords: {', '.join(missing_keywords)}"
                )

        # Skill match recommendations
        skill_match_percentage = skill_analysis.get('skill_match_percentage', 0)
        if skill_match_percentage < 0.4:
            recommendations.append(
                "Increase the overlap between your skills and job requirements"
            )
            recommendations.append(
                "Consider emphasizing transferable skills that relate to the role"
            )

        # Content quality recommendations
        total_resume_skills = skill_analysis.get('total_resume_skills', 0)
        if total_resume_skills < 10:
            recommendations.append(
                "Expand your skills section to include more relevant competencies"
            )

        # Remove duplicates and limit recommendations
        recommendations = list(dict.fromkeys(recommendations))  # Remove duplicates
        return recommendations[:10]  # Limit to top 10

    def calculate_match_score_by_section(self, resume_data: Dict,
                                         job_description: str) -> Dict[str, float]:
        """Calculate match scores for different resume sections"""
        section_scores = {}

        # Get resume sections
        resume_sections = resume_data.get('sections', [])
        resume_text = resume_data.get('text', '')

        # Define section keywords from job description
        jd_keywords = set(self._extract_important_keywords(job_description))

        for section in resume_sections:
            section_lower = section.lower()

            # Extract section text (simplified)
            section_pattern = rf'{re.escape(section)}.*?(?=\n[A-Z]|\n\n|$)'
            section_match = re.search(section_pattern, resume_text, re.IGNORECASE | re.DOTALL)

            if section_match:
                section_text = section_match.group().lower()

                # Count keyword matches in this section
                section_keywords = set(self._extract_important_keywords(section_text))
                matches = len(jd_keywords & section_keywords)
                total_possible = len(jd_keywords)

                section_scores[section] = float(matches / total_possible) if total_possible > 0 else 0.0
            else:
                section_scores[section] = 0.0

        return section_scores

    def analyze_ats_compliance(self, resume_data: Dict) -> Dict[str, Any]:
        """Comprehensive ATS compliance analysis"""
        text = resume_data.get('text', '')

        compliance_checks = {
            'file_format': self._check_file_format(resume_data.get('file_type', '')),
            'standard_sections': self._check_standard_sections(resume_data.get('sections', [])),
            'contact_info': self._check_contact_info(resume_data.get('contact_info', {})),
            'formatting': self._check_formatting_compliance(text),
            'length': self._check_length_compliance(resume_data.get('word_count', 0)),
            'keywords': self._check_keyword_usage(text),
            'readability': self._check_readability_compliance(text)
        }

        # Calculate overall compliance score (ensure floats)
        scores = [float(check.get('score', 0.0)) for check in compliance_checks.values() if isinstance(check, dict)]
        overall_score = float(sum(scores) / len(scores)) if scores else 0.0

        # Generate compliance report
        compliance_report = {
            'overall_score': overall_score,
            'individual_scores': compliance_checks,
            'passed_checks': [k for k, v in compliance_checks.items() if (isinstance(v, dict) and v.get('score', 0) >= 0.8)],
            'failed_checks': [k for k, v in compliance_checks.items() if (isinstance(v, dict) and v.get('score', 0) < 0.6)],
            'grade': self._get_compliance_grade(overall_score)
        }

        # ADDED: ensure serializable before returning
        return to_serializable(compliance_report)

    def _check_file_format(self, file_type: str) -> Dict[str, Any]:
        """Check file format compliance"""
        compliant_formats = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]

        is_compliant = file_type in compliant_formats

        return {
            'score': float(1.0 if is_compliant else 0.5),
            'compliant': is_compliant,
            'message': 'File format is ATS-friendly' if is_compliant else 'Consider using PDF or DOCX format'
        }

    def _check_standard_sections(self, sections: List[str]) -> Dict[str, Any]:
        """Check for standard resume sections"""
        required_sections = ['experience', 'education', 'skills']
        optional_sections = ['summary', 'certifications', 'projects']

        sections_lower = [s.lower() for s in sections]

        required_present = sum(1 for req in required_sections
                               if any(req in sec for sec in sections_lower))
        optional_present = sum(1 for opt in optional_sections
                               if any(opt in sec for sec in sections_lower))

        score = float((required_present / len(required_sections)) * 0.8 +
                      (optional_present / len(optional_sections)) * 0.2) if (len(required_sections) and len(optional_sections)) else 0.0

        return {
            'score': score,
            'required_present': required_present,
            'optional_present': optional_present,
            'message': f'{required_present}/{len(required_sections)} required sections found'
        }

    def _check_contact_info(self, contact_info: Dict) -> Dict[str, Any]:
        """Check contact information completeness"""
        required_fields = ['email', 'phone']
        optional_fields = ['linkedin', 'location']

        required_present = sum(1 for field in required_fields if contact_info.get(field))
        optional_present = sum(1 for field in optional_fields if contact_info.get(field))

        score = float((required_present / len(required_fields)) * 0.8 +
                      (optional_present / len(optional_fields)) * 0.2) if (len(required_fields) and len(optional_fields)) else 0.0

        return {
            'score': score,
            'required_present': required_present,
            'optional_present': optional_present,
            'message': f'{required_present}/{len(required_fields)} required contact fields found'
        }

    def _check_formatting_compliance(self, text: str) -> Dict[str, Any]:
        """Check formatting compliance with ATS standards"""
        issues = 0
        total_checks = 5

        # Check for complex formatting
        if len(re.findall(r'[│┤┐└┴┌┬├─━]', text)) > 5:
            issues += 1

        # Check for excessive special characters
        if len(re.findall(r'[^\w\s\.\,\-\(\)\@\#\$\%\&\*\+\=\/\\\[\]\{\}\|\`\~\"\'\<\>]', text)) > 20:
            issues += 1

        # Check for consistent spacing
        if len(re.findall(r'  +', text)) > 10:
            issues += 1

        # Check for reasonable line lengths (not too many very short lines)
        lines = text.split('\n')
        short_lines = [line for line in lines if len(line.strip()) < 20 and len(line.strip()) > 0]
        if len(lines) > 0 and len(short_lines) > len(lines) * 0.4:
            issues += 1

        # Check for headers/footers indicators
        if re.search(r'page \d+ of \d+|header|footer', text, re.IGNORECASE):
            issues += 1

        score = float(max(0, (total_checks - issues) / total_checks))

        return {
            'score': score,
            'issues_found': issues,
            'message': f'{issues} formatting issues detected' if issues > 0 else 'Formatting looks ATS-friendly'
        }

    def _check_length_compliance(self, word_count: int) -> Dict[str, Any]:
        """Check resume length compliance"""
        if 300 <= word_count <= 800:
            score = 1.0
            message = "Resume length is optimal"
        elif 200 <= word_count < 300 or 800 < word_count <= 1000:
            score = 0.8
            message = "Resume length is acceptable but could be optimized"
        elif 150 <= word_count < 200 or 1000 < word_count <= 1200:
            score = 0.6
            message = "Resume length needs attention"
        else:
            score = 0.4
            message = "Resume length is not optimal for ATS systems"

        return {
            'score': float(score),
            'word_count': int(word_count),
            'message': message
        }

    def _check_keyword_usage(self, text: str) -> Dict[str, Any]:
        """Check appropriate keyword usage"""
        keywords = self._extract_important_keywords(text)
        keyword_count = len(keywords)

        # Calculate keyword density
        words = len(text.split())
        keyword_density = float(keyword_count / words) if words > 0 else 0.0

        if 0.02 <= keyword_density <= 0.08:  # 2-8% keyword density
            score = 1.0
            message = "Keyword usage is well-balanced"
        elif 0.01 <= keyword_density < 0.02 or 0.08 < keyword_density <= 0.12:
            score = 0.7
            message = "Keyword usage could be optimized"
        else:
            score = 0.5
            message = "Keyword usage needs improvement"

        return {
            'score': float(score),
            'keyword_density': keyword_density,
            'total_keywords': int(keyword_count),
            'message': message
        }

    def _check_readability_compliance(self, text: str) -> Dict[str, Any]:
        """Check readability for ATS systems"""
        # Basic readability checks
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {'score': 0.0, 'message': 'No readable content found'}

        words = text.split()
        avg_sentence_length = float(len(words) / len(sentences)) if len(sentences) > 0 else 0.0

        # Optimal sentence length for ATS: 15-25 words
        if 15 <= avg_sentence_length <= 25:
            score = 1.0
            message = "Sentence length is optimal for ATS parsing"
        elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
            score = 0.8
            message = "Sentence length is acceptable"
        else:
            score = 0.6
            message = "Consider adjusting sentence length for better ATS compatibility"

        return {
            'score': float(score),
            'avg_sentence_length': avg_sentence_length,
            'total_sentences': int(len(sentences)),
            'message': message
        }

    def _get_compliance_grade(self, score: float) -> str:
        """Convert compliance score to letter grade"""
        # UPDATED: complete grading scale and ensure correct mapping
        if score >= 0.95:
            return 'A+'
        elif score >= 0.90:
            return 'A'
        elif score >= 0.85:
            return 'A-'
        elif score >= 0.80:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.70:
            return 'B-'
        elif score >= 0.65:
            return 'C+'
        elif score >= 0.60:
            return 'C'
        elif score >= 0.55:
            return 'C-'
        elif score >= 0.50:
            return 'D'
        else:
            return 'F'
