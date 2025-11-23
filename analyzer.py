import re
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import pytesseract
from PIL import Image
import io
import json

class ResumeAnalyzer:
    """Advanced resume analyzer for layout, structure, and ATS compatibility"""
    
    def __init__(self):
        self.ats_friendly_patterns = {
            'standard_fonts': ['Arial', 'Helvetica', 'Calibri', 'Times New Roman', 'Georgia'],
            'sections_required': ['experience', 'education', 'skills'],
            'avoid_elements': ['tables', 'text_boxes', 'graphics', 'columns']
        }
        
        # ATS compliance rules
        self.compliance_rules = {
            'max_columns': 1,
            'min_font_size': 10,
            'max_font_size': 14,
            'avoid_headers_footers': True,
            'simple_formatting': True,
            'standard_section_names': True
        }
    
    def analyze_resume(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive resume analysis"""
        
        analysis_results = {
            'layout_score': self._analyze_layout_compliance(parsed_data),
            'structure_score': self._analyze_structure(parsed_data),
            'content_score': self._analyze_content_quality(parsed_data),
            'keyword_density': self._analyze_keyword_density(parsed_data),
            'sections': self._analyze_sections(parsed_data),
            'issues': self._identify_issues(parsed_data),
            'recommendations': self._generate_recommendations(parsed_data),
            'readability_score': self._calculate_readability(parsed_data),
            'formatting_issues': self._detect_formatting_issues(parsed_data),
            'overall_grade': 'A'  # Will be calculated
        }
        
        # Calculate overall grade
        analysis_results['overall_grade'] = self._calculate_overall_grade(analysis_results)
        
        return analysis_results
    
    def _analyze_layout_compliance(self, parsed_data: Dict) -> float:
        """Analyze ATS layout compliance"""
        score = 1.0
        text = parsed_data.get('text', '')
        
        # Check for multi-column layouts (heuristic)
        lines = text.split('\n')
        short_lines = [line for line in lines if len(line.strip()) < 30 and len(line.strip()) > 5]
        if len(short_lines) > len(lines) * 0.3:  # More than 30% short lines might indicate columns
            score -= 0.2
        
        # Check for excessive formatting characters
        formatting_chars = re.findall(r'[│┤┐└┴┌┬├─━▪▫■□●○◆◇★☆]', text)
        if len(formatting_chars) > 10:
            score -= 0.15
        
        # Check for table-like structures
        table_indicators = re.findall(r'\|.*\||\+[-+]+\+|┌.*┐', text)
        if table_indicators:
            score -= 0.25
        
        # Check for excessive whitespace/formatting
        if len(re.findall(r'\s{5,}', text)) > 20:  # Many large whitespace gaps
            score -= 0.1
        
        # Bonus for standard section headers
        standard_headers = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'SUMMARY']
        found_headers = sum(1 for header in standard_headers if header.lower() in text.lower())
        score += found_headers * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _analyze_structure(self, parsed_data: Dict) -> float:
        """Analyze resume structure and organization"""
        score = 0.0
        sections = parsed_data.get('sections', [])
        
        # Essential sections check
        essential_sections = ['experience', 'education', 'skills']
        for section in essential_sections:
            if any(section in s.lower() for s in sections):
                score += 0.25
        
        # Contact information check
        if parsed_data.get('contact_info'):
            contact = parsed_data['contact_info']
            if contact.get('email'):
                score += 0.1
            if contact.get('phone'):
                score += 0.1
        
        # Logical section order
        text = parsed_data.get('text', '').lower()
        section_positions = {}
        
        for section_name in ['summary', 'experience', 'education', 'skills']:
            match = re.search(rf'\b{section_name}\b', text)
            if match:
                section_positions[section_name] = match.start()
        
        # Check if experience comes before education (common good practice)
        if 'experience' in section_positions and 'education' in section_positions:
            if section_positions['experience'] < section_positions['education']:
                score += 0.05
        
        return min(1.0, score)
    
    def _analyze_content_quality(self, parsed_data: Dict) -> float:
        """Analyze content quality and completeness"""
        score = 0.0
        text = parsed_data.get('text', '')
        
        # Word count adequacy
        word_count = parsed_data.get('word_count', 0)
        if word_count >= 200:
            score += 0.2
        elif word_count >= 150:
            score += 0.15
        elif word_count >= 100:
            score += 0.1
        
        # Action verbs check
        action_verbs = [
            'achieved', 'improved', 'developed', 'created', 'managed', 'led', 'implemented',
            'designed', 'built', 'delivered', 'optimized', 'increased', 'reduced', 'streamlined'
        ]
        
        found_verbs = sum(1 for verb in action_verbs if verb in text.lower())
        score += min(0.2, found_verbs * 0.02)
        
        # Quantified achievements
        quantified_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
            r'\d+(?:,\d{3})*\+?\s*(?:users|customers|clients|people|employees|projects)',
            r'(?:increased|improved|reduced|saved|generated).*?\d+[%$]?'
        ]
        
        quantified_count = 0
        for pattern in quantified_patterns:
            quantified_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        score += min(0.25, quantified_count * 0.05)
        
        # Skills diversity
        skills = parsed_data.get('skills', [])
        if len(skills) >= 5:
            score += 0.15
        elif len(skills) >= 3:
            score += 0.1
        
        # Professional language check (avoid first person)
        first_person_count = len(re.findall(r'\b(?:I|my|me|myself)\b', text, re.IGNORECASE))
        if first_person_count == 0:
            score += 0.1
        elif first_person_count <= 3:
            score += 0.05
        
        # Spelling and grammar heuristics
        common_errors = [
            r'\bteh\b', r'\brecieve\b', r'\boccured\b', r'\bmanagment\b',
            r'\bresponsible\s+of\b', r'\bexperience\s+in\s+working\b'
        ]
        
        error_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in common_errors)
        if error_count == 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_keyword_density(self, parsed_data: Dict) -> Dict[str, float]:
        """Analyze keyword density for different categories"""
        text = parsed_data.get('text', '').lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return {}
        
        keyword_categories = {
            'technical': [
                'software', 'development', 'programming', 'database', 'system', 'application',
                'web', 'mobile', 'cloud', 'api', 'framework', 'algorithm', 'data', 'analysis'
            ],
            'leadership': [
                'team', 'lead', 'manage', 'supervise', 'coordinate', 'direct', 'guide',
                'mentor', 'train', 'delegate', 'organize', 'plan', 'strategy'
            ],
            'business': [
                'revenue', 'profit', 'sales', 'client', 'customer', 'market', 'business',
                'growth', 'strategy', 'optimization', 'efficiency', 'cost', 'budget'
            ]
        }
        
        density_scores = {}
        for category, keywords in keyword_categories.items():
            keyword_count = sum(text.count(keyword) for keyword in keywords)
            density_scores[category] = (keyword_count / word_count) * 100
        
        return density_scores
    
    def _analyze_sections(self, parsed_data: Dict) -> Dict[str, Any]:
        """Detailed analysis of each resume section"""
        sections_analysis = {}
        sections = parsed_data.get('sections', [])
        
        for section in sections:
            section_lower = section.lower()
            sections_analysis[section] = {
                'present': True,
                'ats_friendly': section_lower in ['experience', 'education', 'skills', 'summary', 'certifications'],
                'length_appropriate': True,  # Will be updated with actual analysis
                'quality_score': 0.8  # Placeholder
            }
        
        # Check for missing critical sections
        critical_sections = ['Experience', 'Education', 'Skills']
        for critical in critical_sections:
            if not any(critical.lower() in s.lower() for s in sections):
                sections_analysis[f"Missing_{critical}"] = {
                    'present': False,
                    'ats_friendly': True,
                    'critical': True
                }
        
        return sections_analysis
    
    def _identify_issues(self, parsed_data: Dict) -> List[str]:
        """Identify specific issues with the resume"""
        issues = []
        text = parsed_data.get('text', '')
        
        # Missing contact information
        contact_info = parsed_data.get('contact_info', {})
        if not contact_info.get('email'):
            issues.append("Missing email address")
        if not contact_info.get('phone'):
            issues.append("Missing phone number")
        
        # File format issues
        if parsed_data.get('file_type') not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            issues.append("Unsupported file format - use PDF or DOCX")
        
        # Length issues
        word_count = parsed_data.get('word_count', 0)
        if word_count < 150:
            issues.append("Resume is too short - consider adding more detail")
        elif word_count > 800:
            issues.append("Resume might be too long - consider condensing")
        
        # Formatting issues
        if len(re.findall(r'[│┤┐└┴┌┬├─━]', text)) > 5:
            issues.append("Contains complex formatting that may confuse ATS systems")
        
        # Inconsistent date formatting
        date_formats = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-–]\d{4}|\w+ \d{4}', text)
        if len(set(date_formats)) > 2:
            issues.append("Inconsistent date formatting")
        
        # Missing quantified achievements
        if not re.search(r'\d+[%$]', text):
            issues.append("Consider adding quantified achievements with numbers/percentages")
        
        # Too many bullet points
        bullet_count = len(re.findall(r'[•·▪▫■□]', text))
        if bullet_count > 50:
            issues.append("Too many bullet points - consider consolidating")
        
        return issues
    
    def _generate_recommendations(self, parsed_data: Dict) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        text = parsed_data.get('text', '')
        
        # Skills recommendations
        skills = parsed_data.get('skills', [])
        if len(skills) < 5:
            recommendations.append("Add more relevant technical and soft skills")
        
        # Quantification recommendations
        if not re.search(r'\d+%', text):
            recommendations.append("Add percentage improvements to showcase impact")
        
        if not re.search(r'\$\d+', text):
            recommendations.append("Include monetary values where applicable")
        
        # Action verbs
        action_verb_count = len(re.findall(r'\b(?:achieved|improved|developed|created|managed|led)\b', text, re.IGNORECASE))
        if action_verb_count < 5:
            recommendations.append("Use more strong action verbs to describe your experience")
        
        # Keywords
        keyword_density = self._analyze_keyword_density(parsed_data)
        if keyword_density.get('technical', 0) < 2:
            recommendations.append("Include more industry-specific technical keywords")
        
        # Structure
        sections = parsed_data.get('sections', [])
        if 'summary' not in [s.lower() for s in sections]:
            recommendations.append("Consider adding a professional summary at the top")
        
        # Professional formatting
        if len(re.findall(r'\b(?:I|my|me)\b', text, re.IGNORECASE)) > 5:
            recommendations.append("Remove first-person pronouns for more professional tone")
        
        return recommendations
    
    def _calculate_readability(self, parsed_data: Dict) -> Dict[str, float]:
        """Calculate readability metrics"""
        text = parsed_data.get('text', '')
        
        if not text:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        # Basic readability calculations
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _detect_formatting_issues(self, parsed_data: Dict) -> List[str]:
        """Detect specific formatting issues"""
        issues = []
        text = parsed_data.get('text', '')
        
        # Check for unusual characters
        unusual_chars = re.findall(r'[^\w\s\.,;:!?\-()@#$%&*+=/\\\[\]{}|`~"\'<>]', text)
        if len(unusual_chars) > 10:
            issues.append("Contains unusual characters that may not parse correctly")
        
        # Check for excessive capitalization
        caps_ratio = len(re.findall(r'[A-Z]', text)) / len(text) if text else 0
        if caps_ratio > 0.15:
            issues.append("Excessive use of capital letters")
        
        # Check for inconsistent spacing
        if len(re.findall(r'  +', text)) > 10:
            issues.append("Inconsistent spacing throughout document")
        
        # Check for missing spaces after punctuation
        missing_spaces = re.findall(r'[.,:;!?][A-Za-z]', text)
        if len(missing_spaces) > 3:
            issues.append("Missing spaces after punctuation")
        
        return issues
    
    def _calculate_overall_grade(self, analysis_results: Dict) -> str:
        """Calculate overall grade based on all metrics"""
        scores = [
            analysis_results.get('layout_score', 0),
            analysis_results.get('structure_score', 0),
            analysis_results.get('content_score', 0)
        ]
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.9:
            return 'A+'
        elif avg_score >= 0.8:
            return 'A'
        elif avg_score >= 0.7:
            return 'B'
        elif avg_score >= 0.6:
            return 'C'
        elif avg_score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def analyze_visual_layout(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze visual layout using computer vision (optional feature)"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect text regions
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Simple layout analysis
            height, width = gray.shape
            
            # Divide into sections and analyze density
            sections = {
                'top': gray[:height//3, :],
                'middle': gray[height//3:2*height//3, :],
                'bottom': gray[2*height//3:, :]
            }
            
            layout_analysis = {}
            for section_name, section_img in sections.items():
                # Calculate text density (inverse of white space)
                non_white_pixels = np.sum(section_img < 240)
                total_pixels = section_img.size
                density = non_white_pixels / total_pixels
                
                layout_analysis[f'{section_name}_density'] = density
            
            # Column detection (simplified)
            vertical_projection = np.sum(gray < 240, axis=0)
            peaks = []
            threshold = np.max(vertical_projection) * 0.1
            
            for i in range(1, len(vertical_projection) - 1):
                if (vertical_projection[i] > threshold and 
                    vertical_projection[i] > vertical_projection[i-1] and 
                    vertical_projection[i] > vertical_projection[i+1]):
                    peaks.append(i)
            
            # Estimate number of columns
            if len(peaks) <= 1:
                estimated_columns = 1
            elif len(peaks) <= 3:
                estimated_columns = 2
            else:
                estimated_columns = 3
            
            layout_analysis['estimated_columns'] = estimated_columns
            layout_analysis['ats_friendly_layout'] = estimated_columns <= 1
            
            return layout_analysis
            
        except Exception as e:
            print(f"Visual layout analysis failed: {e}")
            return {'error': 'Could not analyze visual layout'}
    
    def extract_text_with_ocr(self, image_data: bytes) -> str:
        """Extract text using OCR as fallback"""
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""