import re
import json
from typing import Dict, List, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

class ResumeRewriter:
    """AI-powered resume rewriter using Flan-T5 for ATS optimization"""
    
    def __init__(self):
        # Initialize the Flan-T5 model for text rewriting
        try:
            self.model_name = "google/flan-t5-base"
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load Flan-T5 model: {e}")
            self.model_loaded = False
        
        # ATS-optimized writing templates
        self.rewriting_templates = {
            'experience': {
                'prompt_template': """Rewrite this work experience bullet point to be more ATS-friendly and impactful. Use strong action verbs, include quantifiable results where possible, and optimize for keyword matching. Keep it professional and concise.

Original: {text}
Job Description Context: {context}

Rewritten:""",
                'fallback_improvements': [
                    'Start with strong action verbs',
                    'Include quantifiable results',
                    'Use industry-specific keywords',
                    'Remove personal pronouns',
                    'Keep bullets concise (1-2 lines)'
                ]
            },
            'skills': {
                'prompt_template': """Reorganize and optimize this skills section for ATS systems. Group similar skills together, use industry-standard terminology, and ensure keyword compatibility with the job description.

Original Skills: {text}
Job Description Context: {context}

Optimized Skills:""",
                'fallback_improvements': [
                    'Group skills by category',
                    'Use standard skill names',
                    'Include job-relevant keywords',
                    'Remove outdated technologies',
                    'Prioritize in-demand skills'
                ]
            },
            'summary': {
                'prompt_template': """Create a professional summary that is optimized for ATS systems and matches the job requirements. Include key qualifications, years of experience, and relevant skills.

Background Information: {text}
Job Description: {context}

Professional Summary:""",
                'fallback_improvements': [
                    'Lead with years of experience',
                    'Highlight key qualifications',
                    'Include relevant keywords',
                    'Keep to 3-4 sentences',
                    'Focus on value proposition'
                ]
            },
            'education': {
                'prompt_template': """Optimize this education section for ATS parsing. Ensure proper formatting, include relevant coursework or achievements if applicable, and align with job requirements.

Original Education: {text}
Job Requirements: {context}

Optimized Education:""",
                'fallback_improvements': [
                    'Use standard degree names',
                    'Include graduation year',
                    'Add relevant coursework',
                    'Mention academic achievements',
                    'Use consistent formatting'
                ]
            }
        }
        
        # Professional resume templates
        self.resume_templates = {
            'professional': {
                'font_size': 11,
                'font_name': 'Helvetica',
                'margins': (0.75, 0.75, 0.75, 0.75),
                'section_spacing': 0.2,
                'colors': {
                    'header': colors.black,
                    'section_header': colors.black,
                    'text': colors.black
                }
            },
            'modern': {
                'font_size': 10,
                'font_name': 'Helvetica',
                'margins': (0.75, 0.75, 0.75, 0.75),
                'section_spacing': 0.15,
                'colors': {
                    'header': colors.Color(0.2, 0.2, 0.2),
                    'section_header': colors.Color(0.3, 0.3, 0.3),
                    'text': colors.black
                }
            },
            'creative': {
                'font_size': 11,
                'font_name': 'Helvetica-Bold',
                'margins': (0.8, 0.8, 0.8, 0.8),
                'section_spacing': 0.25,
                'colors': {
                    'header': colors.Color(0.1, 0.3, 0.6),
                    'section_header': colors.Color(0.2, 0.4, 0.7),
                    'text': colors.black
                }
            },
            'minimal': {
                'font_size': 11,
                'font_name': 'Times-Roman',
                'margins': (1.0, 1.0, 1.0, 1.0),
                'section_spacing': 0.3,
                'colors': {
                    'header': colors.black,
                    'section_header': colors.black,
                    'text': colors.black
                }
            }
        }
    
    def rewrite_resume(self, resume_data: Dict, job_description: str, 
                      ats_scores: Dict, sections: List[str] = None, 
                      style: str = 'professional') -> Dict[str, Any]:
        """Rewrite resume sections using AI for ATS optimization"""
        
        if sections is None:
            sections = ['experience', 'skills', 'summary']
        
        rewritten_resume = resume_data.copy()
        improvements = []
        
        # Extract and rewrite each requested section
        for section in sections:
            section_lower = section.lower()
            
            if section_lower in self.rewriting_templates:
                # Extract section content
                section_content = self._extract_section_content(
                    resume_data.get('text', ''), section_lower
                )
                
                if section_content:
                    # Rewrite the section
                    rewritten_content = self._rewrite_section(
                        section_content, job_description, section_lower
                    )
                    
                    if rewritten_content:
                        rewritten_resume[f'{section_lower}_optimized'] = rewritten_content
                        improvements.append(f"Optimized {section} section for ATS compatibility")
                    else:
                        # Use fallback improvements
                        fallback_content = self._apply_fallback_improvements(
                            section_content, section_lower
                        )
                        rewritten_resume[f'{section_lower}_optimized'] = fallback_content
                        improvements.append(f"Applied manual optimizations to {section} section")
        
        # Generate overall optimized text
        optimized_text = self._combine_optimized_sections(rewritten_resume, resume_data)
        rewritten_resume['text'] = optimized_text
        rewritten_resume['improvements'] = improvements
        rewritten_resume['optimization_score'] = self._calculate_optimization_improvement(
            ats_scores, rewritten_resume
        )
        
        return rewritten_resume
    
    def _extract_section_content(self, text: str, section: str) -> str:
        """Extract content from a specific resume section"""
        
        # Define section header patterns
        section_patterns = {
            'experience': r'(?i)(?:work\s+)?experience|employment|professional\s+background',
            'skills': r'(?i)(?:technical\s+)?skills|competencies|proficiencies',
            'summary': r'(?i)(?:professional\s+)?summary|profile|objective|about',
            'education': r'(?i)education|academic\s+background|qualifications'
        }
        
        if section not in section_patterns:
            return ""
        
        pattern = section_patterns[section]
        
        # Find the section header
        section_match = re.search(f'({pattern})', text, re.IGNORECASE)
        
        if not section_match:
            return ""
        
        start_pos = section_match.end()
        
        # Find the next section or end of document
        next_section_pattern = r'\n\s*(?:[A-Z][A-Za-z\s]{5,})\s*\n'
        next_section_match = re.search(next_section_pattern, text[start_pos:])
        
        if next_section_match:
            end_pos = start_pos + next_section_match.start()
            section_content = text[start_pos:end_pos]
        else:
            section_content = text[start_pos:]
        
        # Clean up the content
        section_content = section_content.strip()
        
        # Remove excessive whitespace
        section_content = re.sub(r'\n\s*\n', '\n', section_content)
        section_content = re.sub(r' +', ' ', section_content)
        
        return section_content
    
    def _rewrite_section(self, content: str, job_description: str, section_type: str) -> str:
        """Rewrite a section using AI model"""
        
        if not self.model_loaded or not content.strip():
            return ""
        
        try:
            template = self.rewriting_templates[section_type]['prompt_template']
            
            # Prepare the prompt
            prompt = template.format(
                text=content[:500],  # Limit input length
                context=job_description[:300]  # Limit context length
            )
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            rewritten_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            rewritten_text = self._clean_ai_output(rewritten_text, section_type)
            
            return rewritten_text
            
        except Exception as e:
            print(f"AI rewriting failed for {section_type}: {e}")
            return ""
    
    def _clean_ai_output(self, text: str, section_type: str) -> str:
        """Clean and validate AI-generated output"""
        
        # Remove the original prompt if it's repeated
        text = re.sub(r'.*?(?:Rewritten:|Optimized.*?:|Professional Summary:)', '', text, flags=re.IGNORECASE)
        
        # Clean up formatting
        text = text.strip()
        
        # Ensure proper bullet points for experience/skills
        if section_type in ['experience', 'skills']:
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('•') and not line.startswith('-'):
                    line = f"• {line}"
                cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Ensure reasonable length
        if len(text) > 1000:
            sentences = text.split('.')
            text = '. '.join(sentences[:5]) + '.'
        
        return text.strip()
    
    def _apply_fallback_improvements(self, content: str, section_type: str) -> str:
        """Apply manual improvements when AI rewriting fails"""
        
        improvements = self.rewriting_templates[section_type]['fallback_improvements']
        improved_content = content
        
        if section_type == 'experience':
            improved_content = self._improve_experience_section(content)
        elif section_type == 'skills':
            improved_content = self._improve_skills_section(content)
        elif section_type == 'summary':
            improved_content = self._improve_summary_section(content)
        elif section_type == 'education':
            improved_content = self._improve_education_section(content)
        
        return improved_content
    
    def _improve_experience_section(self, content: str) -> str:
        """Apply manual improvements to experience section"""
        
        lines = content.split('\n')
        improved_lines = []
        
        action_verbs = [
            'Achieved', 'Improved', 'Developed', 'Created', 'Managed', 'Led',
            'Implemented', 'Designed', 'Built', 'Delivered', 'Optimized',
            'Increased', 'Reduced', 'Streamlined', 'Collaborated', 'Coordinated'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Ensure bullet points
            if not line.startswith('•') and not line.startswith('-'):
                line = f"• {line}"
            
            # Replace weak verbs with strong action verbs
            line = re.sub(r'^\s*•?\s*(?:Was\s+)?responsible\s+for', '• Managed', line, flags=re.IGNORECASE)
            line = re.sub(r'^\s*•?\s*Worked\s+on', '• Developed', line, flags=re.IGNORECASE)
            line = re.sub(r'^\s*•?\s*Helped\s+with', '• Supported', line, flags=re.IGNORECASE)
            
            # Remove first person pronouns
            line = re.sub(r'\bI\b', '', line, flags=re.IGNORECASE)
            line = re.sub(r'\bmy\b', 'the', line, flags=re.IGNORECASE)
            line = re.sub(r'\bme\b', '', line, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            line = re.sub(r' +', ' ', line).strip()
            
            improved_lines.append(line)
        
        return '\n'.join(improved_lines)
    
    def _improve_skills_section(self, content: str) -> str:
        """Apply manual improvements to skills section"""
        
        # Extract individual skills
        skills_text = content.lower()
        skills = re.split(r'[,•\n\-\|]', skills_text)
        
        # Clean and standardize skills
        cleaned_skills = []
        skill_categories = {
            'Programming Languages': [],
            'Frameworks & Libraries': [],
            'Databases': [],
            'Tools & Technologies': [],
            'Soft Skills': []
        }
        
        # Categorize skills
        programming_keywords = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go']
        framework_keywords = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']
        database_keywords = ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle']
        tool_keywords = ['git', 'docker', 'kubernetes', 'aws', 'azure', 'linux', 'jenkins']
        soft_keywords = ['leadership', 'communication', 'teamwork', 'problem', 'management']
        
        for skill in skills:
            skill = skill.strip().title()
            if len(skill) > 1 and len(skill) < 30:
                skill_lower = skill.lower()
                
                if any(prog in skill_lower for prog in programming_keywords):
                    skill_categories['Programming Languages'].append(skill)
                elif any(fw in skill_lower for fw in framework_keywords):
                    skill_categories['Frameworks & Libraries'].append(skill)
                elif any(db in skill_lower for db in database_keywords):
                    skill_categories['Databases'].append(skill)
                elif any(tool in skill_lower for tool in tool_keywords):
                    skill_categories['Tools & Technologies'].append(skill)
                elif any(soft in skill_lower for soft in soft_keywords):
                    skill_categories['Soft Skills'].append(skill)
                else:
                    cleaned_skills.append(skill)
        
        # Organize output
        organized_output = []
        
        for category, items in skill_categories.items():
            if items:
                organized_output.append(f"{category}:")
                organized_output.append(", ".join(items))
                organized_output.append("")
        
        if cleaned_skills:
            organized_output.append("Other Skills:")
            organized_output.append(", ".join(cleaned_skills))
        
        return '\n'.join(organized_output)
    
    def _improve_summary_section(self, content: str) -> str:
        """Apply manual improvements to summary section"""
        
        # Remove first person pronouns
        improved = re.sub(r'\bI am\b', 'Professional with', content, flags=re.IGNORECASE)
        improved = re.sub(r'\bI have\b', 'Possessing', improved, flags=re.IGNORECASE)
        improved = re.sub(r'\bI\b', '', improved, flags=re.IGNORECASE)
        improved = re.sub(r'\bmy\b', '', improved, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        improved = re.sub(r' +', ' ', improved)
        
        # Ensure it starts with years of experience if mentioned
        years_match = re.search(r'(\d+)[\+]?\s*years?', improved, re.IGNORECASE)
        if years_match:
            years = years_match.group(0)
            improved = f"Experienced professional with {years} " + improved
        
        return improved.strip()
    
    def _improve_education_section(self, content: str) -> str:
        """Apply manual improvements to education section"""
        
        lines = content.split('\n')
        improved_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Standardize degree formats
            line = re.sub(r'\bbs\b', 'Bachelor of Science', line, flags=re.IGNORECASE)
            line = re.sub(r'\bba\b', 'Bachelor of Arts', line, flags=re.IGNORECASE)
            line = re.sub(r'\bms\b', 'Master of Science', line, flags=re.IGNORECASE)
            line = re.sub(r'\bmba\b', 'Master of Business Administration', line, flags=re.IGNORECASE)
            
            improved_lines.append(line)
        
        return '\n'.join(improved_lines)
    
    def _combine_optimized_sections(self, rewritten_resume: Dict, original_resume: Dict) -> str:
        """Combine optimized sections into a complete resume text"""
        
        sections = []
        
        # Contact Information
        contact_info = original_resume.get('contact_info', {})
        if contact_info:
            contact_lines = []
            if contact_info.get('email'):
                contact_lines.append(f"Email: {contact_info['email']}")
            if contact_info.get('phone'):
                contact_lines.append(f"Phone: {contact_info['phone']}")
            if contact_info.get('linkedin'):
                contact_lines.append(f"LinkedIn: {contact_info['linkedin']}")
            if contact_info.get('location'):
                contact_lines.append(f"Location: {contact_info['location']}")
            
            if contact_lines:
                sections.append("CONTACT INFORMATION")
                sections.extend(contact_lines)
                sections.append("")
        
        # Professional Summary
        if 'summary_optimized' in rewritten_resume:
            sections.append("PROFESSIONAL SUMMARY")
            sections.append(rewritten_resume['summary_optimized'])
            sections.append("")
        
        # Experience
        if 'experience_optimized' in rewritten_resume:
            sections.append("PROFESSIONAL EXPERIENCE")
            sections.append(rewritten_resume['experience_optimized'])
            sections.append("")
        
        # Skills
        if 'skills_optimized' in rewritten_resume:
            sections.append("TECHNICAL SKILLS")
            sections.append(rewritten_resume['skills_optimized'])
            sections.append("")
        
        # Education
        if 'education_optimized' in rewritten_resume:
            sections.append("EDUCATION")
            sections.append(rewritten_resume['education_optimized'])
            sections.append("")
        elif original_resume.get('education'):
            sections.append("EDUCATION")
            education_text = '\n'.join([f"• {edu.get('degree', '')} {edu.get('field', '')}" 
                                       for edu in original_resume['education']])
            sections.append(education_text)
            sections.append("")
        
        return '\n'.join(sections)
    
    def _calculate_optimization_improvement(self, original_scores: Dict, 
                                          rewritten_resume: Dict) -> float:
        """Calculate the improvement in optimization after rewriting"""
        
        # This is a simplified calculation
        # In a real implementation, you'd re-run the scoring algorithm
        
        improvements_count = len(rewritten_resume.get('improvements', []))
        base_improvement = min(0.15, improvements_count * 0.05)  # Max 15% improvement
        
        original_score = original_scores.get('final_score', 0.5)
        estimated_new_score = min(1.0, original_score + base_improvement)
        
        return estimated_new_score
    
    def generate_pdf(self, optimized_resume: Dict, template_style: str = 'professional') -> str:
        """Generate a professional PDF from the optimized resume"""
        
        filename = f"optimized_resume_{template_style}.pdf"
        template_config = self.resume_templates.get(template_style, self.resume_templates['professional'])
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            topMargin=template_config['margins'][0] * inch,
            bottomMargin=template_config['margins'][1] * inch,
            leftMargin=template_config['margins'][2] * inch,
            rightMargin=template_config['margins'][3] * inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=template_config['colors']['header'],
            fontName=template_config['font_name']
        )
        
        section_header_style = ParagraphStyle(
            'CustomSectionHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            textColor=template_config['colors']['section_header'],
            fontName=template_config['font_name'] + '-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=template_config['font_size'],
            spaceAfter=3,
            textColor=template_config['colors']['text'],
            fontName=template_config['font_name']
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=body_style,
            leftIndent=20,
            bulletIndent=10
        )
        
        # Build the document
        story = []
        
        # Add content from optimized resume
        resume_text = optimized_resume.get('text', '')
        sections = resume_text.split('\n\n')
        
        current_section_header = None
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.split('\n')
            first_line = lines[0].strip()
            
            # Check if this is a section header (all caps, common section names)
            if (first_line.isupper() and 
                any(keyword in first_line for keyword in ['CONTACT', 'SUMMARY', 'EXPERIENCE', 'SKILLS', 'EDUCATION'])):
                
                current_section_header = first_line
                story.append(Paragraph(first_line, section_header_style))
                
                # Process remaining lines in the section
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        if line.startswith('•') or line.startswith('-'):
                            # Bullet point
                            clean_line = line[1:].strip()
                            story.append(Paragraph(f"• {clean_line}", bullet_style))
                        else:
                            # Regular text
                            story.append(Paragraph(line, body_style))
            else:
                # Regular content
                for line in lines:
                    line = line.strip()
                    if line:
                        if line.startswith('•') or line.startswith('-'):
                            clean_line = line[1:].strip()
                            story.append(Paragraph(f"• {clean_line}", bullet_style))
                        else:
                            story.append(Paragraph(line, body_style))
            
            # Add spacing between sections
            story.append(Spacer(1, template_config['section_spacing'] * inch))
        
        # Build the PDF
        try:
            doc.build(story)
            return filename
        except Exception as e:
            print(f"PDF generation failed: {e}")
            return ""
    
    def generate_word_document(self, optimized_resume: Dict, template_style: str = 'professional') -> str:
        """Generate a Word document from the optimized resume"""
        
        try:
            from docx import Document
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            filename = f"optimized_resume_{template_style}.docx"
            doc = Document()
            
            # Set margins
            section = doc.sections[0]
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
            
            # Process resume content
            resume_text = optimized_resume.get('text', '')
            sections = resume_text.split('\n\n')
            
            for section in sections:
                if not section.strip():
                    continue
                
                lines = section.split('\n')
                first_line = lines[0].strip()
                
                # Check for section headers
                if (first_line.isupper() and 
                    any(keyword in first_line for keyword in ['CONTACT', 'SUMMARY', 'EXPERIENCE', 'SKILLS', 'EDUCATION'])):
                    
                    # Add section header
                    header = doc.add_heading(first_line, level=2)
                    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    
                    # Add content
                    for line in lines[1:]:
                        line = line.strip()
                        if line:
                            p = doc.add_paragraph(line)
                            if line.startswith('•') or line.startswith('-'):
                                p.style = 'List Bullet'
                else:
                    # Regular content
                    for line in lines:
                        line = line.strip()
                        if line:
                            p = doc.add_paragraph(line)
                            if line.startswith('•') or line.startswith('-'):
                                p.style = 'List Bullet'
            
            # Save document
            doc.save(filename)
            return filename
            
        except ImportError:
            print("python-docx not available, skipping Word document generation")
            return ""
        except Exception as e:
            print(f"Word document generation failed: {e}")
            return ""
    
    def create_resume_templates(self) -> Dict[str, str]:
        """Create template files for different resume styles"""
        
        templates = {}
        
        for style_name, config in self.resume_templates.items():
            template_content = f"""
# {style_name.title()} Resume Template

**Configuration:**
- Font: {config['font_name']}
- Font Size: {config['font_size']}pt
- Margins: {config['margins']} inches
- Section Spacing: {config['section_spacing']} inches

**Sections:**
1. Contact Information
2. Professional Summary
3. Professional Experience
4. Technical Skills
5. Education
6. Certifications (optional)
7. Projects (optional)

**ATS Optimization Features:**
- Single column layout
- Standard section headers
- Clean formatting
- Keyword optimization
- Quantified achievements
- Action verb usage
"""
            
            templates[style_name] = template_content
        
        return templates