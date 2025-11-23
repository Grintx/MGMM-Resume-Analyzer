import re
import spacy
import pdfplumber
import docx
import json
from typing import Dict, List, Any
import io

class ResumeParser:
    """Resume parser for PDF and DOCX files with NER capabilities"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define resume sections patterns
        self.section_patterns = {
            'education': r'(?i)\b(?:education|academic|qualification|degree|university|college|school)\b',
            'experience': r'(?i)\b(?:experience|employment|work|career|professional|job)\b',
            'skills': r'(?i)\b(?:skills|technical|competenc|abilit|proficien)\b',
            'projects': r'(?i)\b(?:projects|portfolio|work samples)\b',
            'certifications': r'(?i)\b(?:certification|certificate|license|credential)\b',
            'awards': r'(?i)\b(?:awards|honors|achievement|recognition)\b',
            'summary': r'(?i)\b(?:summary|objective|profile|about)\b'
        }
        
        # Define skill categories and keywords
        self.skill_keywords = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'bootstrap'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 
                'sqlite', 'cassandra', 'dynamodb'
            ],
            'tools': [
                'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 
                'linux', 'unix', 'tableau', 'power bi', 'jira', 'confluence'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem-solving', 
                'project management', 'analytical', 'creative', 'adaptable'
            ]
        }
    
    def parse_resume(self, file) -> Dict[str, Any]:
        """Parse uploaded resume file and extract structured information"""
        
        # Determine file type and extract text
        if file.type == "application/pdf":
            text = self._extract_pdf_text(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = self._extract_docx_text(file)
        else:
            raise ValueError("Unsupported file type")
        
        # Parse the text
        parsed_data = {
            'filename': file.name,
            'file_type': file.type,
            'text': text,
            'contact_info': self._extract_contact_info(text),
            'sections': self._identify_sections(text),
            'skills': self._extract_skills(text),
            'entities': self._extract_entities(text),
            'education': self._extract_education(text),
            'experience': self._extract_experience(text),
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        return parsed_data
    
    def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            text = "Error: Could not extract text from PDF"
        
        return text.strip()
    
    def _extract_docx_text(self, file) -> str:
        """Extract text from DOCX using python-docx"""
        text = ""
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
                
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
                    
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            text = "Error: Could not extract text from DOCX"
        
        return text.strip()
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns"""
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone pattern (various formats)
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890, 123.456.7890
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'  # International
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info['phone'] = phones[0]
                break
        
        # LinkedIn pattern
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)([A-Za-z0-9_-]+)'
        linkedin = re.findall(linkedin_pattern, text.lower())
        if linkedin:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin[0]}"
        
        # GitHub pattern
        github_pattern = r'(?:github\.com/)([A-Za-z0-9_-]+)'
        github = re.findall(github_pattern, text.lower())
        if github:
            contact_info['github'] = f"github.com/{github[0]}"
        
        # Location (simple pattern for city, state)
        location_pattern = r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b'
        locations = re.findall(location_pattern, text)
        if locations:
            contact_info['location'] = locations[0]
        
        return contact_info
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify resume sections using pattern matching"""
        sections_found = []
        
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, text):
                sections_found.append(section_name.title())
        
        return sections_found
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical and soft skills"""
        text_lower = text.lower()
        skills_found = set()
        
        # Extract skills from predefined categories
        for category, skills_list in self.skill_keywords.items():
            for skill in skills_list:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    skills_found.add(skill.title())
        
        # Extract skills from skills section if present
        skills_section_match = re.search(
            r'(?i)skills?\s*:?\s*(.*?)(?=\n\s*[A-Z]|\n\s*\n|$)', 
            text, 
            re.DOTALL
        )
        
        if skills_section_match:
            skills_text = skills_section_match.group(1)
            # Split by common delimiters
            skill_items = re.split(r'[,•·\n\-\|]', skills_text)
            for skill in skill_items:
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 30:
                    skills_found.add(skill.title())
        
        return list(skills_found)
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities (cities, countries)
            'DATE': [],
            'MONEY': [],
            'CARDINAL': []  # Numbers
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.strip())
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r'\b(?:Bachelor|B\.?[ASs]\.?|BS|BA|BE|BTech|B\.Tech)\b.*?(?:in|of)?\s*([A-Za-z\s&]+)',
            r'\b(?:Master|M\.?[ASs]\.?|MS|MA|ME|MTech|M\.Tech|MBA)\b.*?(?:in|of)?\s*([A-Za-z\s&]+)',
            r'\b(?:Ph\.?D\.?|PhD|Doctorate)\b.*?(?:in|of)?\s*([A-Za-z\s&]+)',
            r'\b(?:Associate|A\.?[ASs]\.?|AA|AS)\b.*?(?:in|of)?\s*([A-Za-z\s&]+)'
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                degree_info = {
                    'degree': match.group(0).strip(),
                    'field': match.group(1).strip() if len(match.groups()) > 0 else '',
                    'context': text[max(0, match.start()-50):match.end()+50].strip()
                }
                education.append(degree_info)
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience information"""
        experience = []
        
        # Look for job titles with common patterns
        job_patterns = [
            r'\b(?:Senior|Jr\.|Junior|Lead|Principal|Staff|Chief)?\s*(?:Software|Data|Full[- ]Stack|Front[- ]End|Back[- ]End|DevOps|Product|Project|Marketing|Sales|Business|Financial|Operations|Human Resources|Quality Assurance|QA)\s*(?:Engineer|Developer|Analyst|Manager|Director|Coordinator|Specialist|Consultant|Architect|Scientist|Administrator)\b',
            r'\b(?:CEO|CTO|CFO|COO|VP|Vice President|President|Director|Manager)\b',
            r'\b(?:Intern|Internship|Co-op|Trainee)\b'
        ]
        
        for pattern in job_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                job_info = {
                    'title': match.group(0).strip(),
                    'context': text[max(0, match.start()-50):match.end()+100].strip()
                }
                
                # Try to extract company and dates from context
                context = job_info['context']
                
                # Look for company names (capitalized words near the job title)
                company_pattern = r'\b([A-Z][a-zA-Z\s&\.]{2,30}(?:Inc|LLC|Corp|Ltd|Company|Co\.|Technologies|Tech|Systems|Solutions|Services|Group)?)\b'
                companies = re.findall(company_pattern, context)
                if companies:
                    job_info['company'] = companies[0].strip()
                
                # Look for date ranges
                date_pattern = r'\b(\d{4})\s*[-–—]\s*(\d{4}|present|current)\b'
                dates = re.findall(date_pattern, context, re.IGNORECASE)
                if dates:
                    job_info['duration'] = f"{dates[0][0]} - {dates[0][1]}"
                
                experience.append(job_info)
        
        return experience
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\-\(\)\@\#\+]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_resume(self, text: str) -> Dict[str, str]:
        """Segment resume into different sections"""
        sections = {}
        
        # Try to split by section headers
        section_splits = []
        for section_name, pattern in self.section_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                section_splits.append({
                    'name': section_name,
                    'start': match.start(),
                    'header': match.group()
                })
        
        # Sort by position in text
        section_splits.sort(key=lambda x: x['start'])
        
        # Extract content for each section
        for i, section in enumerate(section_splits):
            start_pos = section['start']
            
            # Find end position (start of next section or end of text)
            if i + 1 < len(section_splits):
                end_pos = section_splits[i + 1]['start']
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            # Clean up the section text
            # Remove the header from the content
            header_end = section_text.find('\n')
            if header_end > 0:
                section_content = section_text[header_end:].strip()
            else:
                section_content = section_text
            
            sections[section['name']] = section_content
        
        return sections
    
    def extract_achievements(self, text: str) -> List[str]:
        """Extract achievement statements and quantified results"""
        achievements = []
        
        # Patterns for achievements and quantified results
        achievement_patterns = [
            r'(?:achieved|accomplished|delivered|improved|increased|decreased|reduced|saved|generated|managed|led|developed|created|implemented|designed|built|launched)\s+[^.!?]*?(?:\d+[%$]?|\d+(?:,\d{3})*|\d+\.\d+[%$]?)[^.!?]*[.!?]',
            r'[^.!?]*?(?:\d+[%$]?|\d+(?:,\d{3})*|\d+\.\d+[%$]?)[^.!?]*?(?:increase|decrease|improvement|growth|reduction|savings|revenue|profit|efficiency|performance)[^.!?]*[.!?]',
            r'(?:responsible for|resulted in|contributed to)[^.!?]*?(?:\d+[%$]?|\d+(?:,\d{3})*|\d+\.\d+[%$]?)[^.!?]*[.!?]'
        ]
        
        for pattern in achievement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            achievements.extend([match.strip() for match in matches])
        
        # Remove duplicates and clean
        achievements = list(set(achievements))
        achievements = [ach for ach in achievements if len(ach) > 20 and len(ach) < 200]
        
        return achievements[:10]  # Return top 10
    
    def validate_resume_structure(self, parsed_data: Dict) -> Dict[str, Any]:
        """Validate resume structure and completeness"""
        validation_results = {
            'has_contact_info': bool(parsed_data.get('contact_info')),
            'has_experience': 'experience' in parsed_data.get('sections', []),
            'has_education': 'education' in parsed_data.get('sections', []),
            'has_skills': bool(parsed_data.get('skills')),
            'word_count_adequate': parsed_data.get('word_count', 0) >= 200,
            'completeness_score': 0.0,
            'missing_elements': []
        }
        
        # Calculate completeness score
        score = 0
        total_checks = 5
        
        if validation_results['has_contact_info']:
            score += 1
        else:
            validation_results['missing_elements'].append('Contact Information')
            
        if validation_results['has_experience']:
            score += 1
        else:
            validation_results['missing_elements'].append('Work Experience')
            
        if validation_results['has_education']:
            score += 1
        else:
            validation_results['missing_elements'].append('Education')
            
        if validation_results['has_skills']:
            score += 1
        else:
            validation_results['missing_elements'].append('Skills Section')
            
        if validation_results['word_count_adequate']:
            score += 1
        else:
            validation_results['missing_elements'].append('Sufficient Content (200+ words)')
        
        validation_results['completeness_score'] = score / total_checks
        
        return validation_results