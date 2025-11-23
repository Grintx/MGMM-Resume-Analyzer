import streamlit as st
import json
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from parser import ResumeParser
from analyzer import ResumeAnalyzer
from scorer import ATSScorer
from rewriter import ResumeRewriter
import base64
import os

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_db():
    """Initialize SQLite database for storing results"""
    conn = sqlite3.connect('resume_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TIMESTAMP,
            ats_score REAL,
            layout_score REAL,
            tfidf_score REAL,
            sbert_score REAL,
            missing_skills TEXT,
            extracted_entities TEXT,
            analysis_results TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'parsed_resume' not in st.session_state:
        st.session_state.parsed_resume = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'ats_scores' not in st.session_state:
        st.session_state.ats_scores = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""

# Utility functions
def create_download_link(file_path, filename):
    """Create a download link for files"""
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:file/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def display_ats_score(score):
    """Display ATS score with color coding"""
    if score >= 0.8:
        color = "green"
        status = "Excellent"
    elif score >= 0.6:
        color = "orange"
        status = "Good"
    elif score >= 0.4:
        color = "yellow"
        status = "Fair"
    else:
        color = "red"
        status = "Poor"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center">
            <h2 style="color: {color}">ATS Score: {score:.1%}</h2>
            <h4 style="color: {color}">{status}</h4>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    init_db()
    init_session_state()
    
    # Title and description
    st.title("**Multi-Granularity Multi-Modal Resume Analyzer**")
    st.markdown("""
     Optimize your resume for ATS systems and specific job roles.
    
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox("Select Feature:", [
            "Resume Upload & Analysis",
            "Job Matching",
            "Resume Optimization",
            "Analytics Dashboard"
        ])
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Display some quick stats
        conn = sqlite3.connect('resume_data.db')
        total_resumes = pd.read_sql_query("SELECT COUNT(*) as count FROM resume_analysis", conn).iloc[0]['count']
        conn.close()
        
        st.metric("Total Resumes Analyzed", total_resumes)
    
    # Main content based on selected page
    if page == "Resume Upload & Analysis":
        resume_upload_page()
    elif page == "Job Matching":
        job_matching_page()
    elif page == "Resume Optimization":
        resume_optimization_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()

def resume_upload_page():
    """Resume upload and analysis page"""
    st.header("📄 Resume Upload & Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx'],
        help="Upload your resume in PDF or DOCX format"
    )
    
    if uploaded_file is not None:
        # Parse resume
        with st.spinner("Parsing resume..."):
            parser = ResumeParser()
            parsed_data = parser.parse_resume(uploaded_file)
            st.session_state.parsed_resume = parsed_data
        
        # Analyze resume
        with st.spinner("Analyzing resume structure and content..."):
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(parsed_data)
            st.session_state.analysis_results = analysis_results
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Extracted Information")
            
            # Basic info
            if parsed_data.get('contact_info'):
                st.write("**Contact Information:**")
                for key, value in parsed_data['contact_info'].items():
                    if value:
                        st.write(f"- {key.title()}: {value}")
            
            # Skills
            if parsed_data.get('skills'):
                st.write("**Skills Extracted:**")
                skills = parsed_data['skills'][:10]  # Show first 10
                st.write(", ".join(skills))
                if len(parsed_data['skills']) > 10:
                    st.write(f"... and {len(parsed_data['skills']) - 10} more")
        
        with col2:
            st.subheader("🔍 Structure Analysis")
            
            # Layout score
            layout_score = analysis_results.get('layout_score', 0.5)
            st.metric("Layout Compliance", f"{layout_score:.1%}")
            
            # Issues found
            if analysis_results.get('issues'):
                st.write("**Issues Found:**")
                for issue in analysis_results['issues']:
                    st.warning(f"⚠️ {issue}")
            
            # Sections detected
            if analysis_results.get('sections'):
                st.write("**Sections Detected:**")
                for section in analysis_results['sections']:
                    st.success(f"✅ {section}")
        
        # Text preview
        st.subheader("📝 Parsed Text Preview")
        with st.expander("View extracted text"):
            st.text_area("Resume Text", parsed_data.get('text', ''), height=300, disabled=True)

def job_matching_page():
    """Job matching and ATS scoring page"""
    st.header("🎯 Job Description Matching")
    
    if st.session_state.parsed_resume is None:
        st.warning("Please upload and analyze a resume first!")
        return
    
    # Job description input
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        value=st.session_state.job_description,
        height=200,
        placeholder="Copy and paste the job description you want to match against..."
    )
    
    if st.button("🔍 Analyze Match", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description!")
            return
        
        st.session_state.job_description = job_description
        
        with st.spinner("Computing ATS scores and job match..."):
            scorer = ATSScorer()
            scores = scorer.compute_ats_score(
                st.session_state.parsed_resume,
                job_description,
                st.session_state.analysis_results
            )
            st.session_state.ats_scores = scores
        
        # Display ATS score
        display_ats_score(scores['final_score'])
        
        # Detailed breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Semantic Similarity",
                f"{scores['sbert_score']:.1%}",
                help="SBERT embedding similarity"
            )
        
        with col2:
            st.metric(
                "Keyword Match",
                f"{scores['tfidf_score']:.1%}",
                help="TF-IDF keyword overlap"
            )
        
        with col3:
            st.metric(
                "Layout Score",
                f"{scores['layout_score']:.1%}",
                help="ATS-friendly formatting"
            )
        
        # Skill gap analysis
        st.subheader("📊 Skill Gap Analysis")
        
        if scores.get('missing_skills'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing Skills:**")
                for skill in scores['missing_skills'][:10]:
                    st.error(f"❌ {skill}")
            
            with col2:
                st.write("**Matching Skills:**")
                for skill in scores.get('matching_skills', [])[:10]:
                    st.success(f"✅ {skill}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = scores.get('recommendations', [])
        for rec in recommendations:
            st.info(f"💡 {rec}")
        
        # Save to database
        save_analysis_to_db(scores)

def resume_optimization_page():
    """Resume optimization and rewriting page"""
    st.header("✨ AI-Powered Resume Optimization")
    
    if st.session_state.parsed_resume is None:
        st.warning("Please upload and analyze a resume first!")
        return
    
    if st.session_state.ats_scores is None:
        st.warning("Please run job matching analysis first!")
        return
    
    st.subheader("🤖 AI Resume Rewriter")
    st.write("Generate an ATS-optimized version of your resume using AI.")
    
    # Rewriting options
    col1, col2 = st.columns(2)
    
    with col1:
        rewrite_sections = st.multiselect(
            "Select sections to optimize:",
            ["Experience", "Skills", "Summary", "Education"],
            default=["Experience", "Skills"]
        )
    
    with col2:
        template_style = st.selectbox(
            "Choose template style:",
            ["Professional", "Modern", "Creative", "Minimal"]
        )
    
    if st.button("🚀 Generate Optimized Resume", type="primary"):
        with st.spinner("AI is rewriting your resume..."):
            rewriter = ResumeRewriter()
            
            # Rewrite selected sections
            optimized_resume = rewriter.rewrite_resume(
                st.session_state.parsed_resume,
                st.session_state.job_description,
                st.session_state.ats_scores,
                sections=rewrite_sections,
                style=template_style
            )
            
            # Generate PDF
            pdf_path = rewriter.generate_pdf(optimized_resume, template_style)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Original Resume")
                st.text_area("Original", st.session_state.parsed_resume.get('text', ''), height=400, disabled=True)
            
            with col2:
                st.subheader("✨ Optimized Resume")
                st.text_area("Optimized", optimized_resume.get('text', ''), height=400, disabled=True)
            
            # Download button
            if os.path.exists(pdf_path):
                st.success("✅ Optimized resume generated successfully!")
                download_link = create_download_link(pdf_path, "optimized_resume.pdf")
                st.markdown(download_link, unsafe_allow_html=True)
            
            # Improvement summary
            st.subheader("📈 Improvements Made")
            improvements = optimized_resume.get('improvements', [])
            for improvement in improvements:
                st.info(f"✨ {improvement}")

def analytics_dashboard_page():
    """Analytics and history dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Load data from database
    conn = sqlite3.connect('resume_data.db')
    df = pd.read_sql_query("""
        SELECT * FROM resume_analysis 
        ORDER BY upload_date DESC
    """, conn)
    conn.close()
    
    if df.empty:
        st.info("No analysis data available yet. Upload and analyze some resumes first!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(df))
    
    with col2:
        avg_score = df['ats_score'].mean() if 'ats_score' in df else 0
        st.metric("Average ATS Score", f"{avg_score:.1%}")
    
    with col3:
        high_scores = (df['ats_score'] >= 0.7).sum() if 'ats_score' in df else 0
        st.metric("High-Quality Resumes", high_scores)
    
    with col4:
        recent = df[df['upload_date'] >= (datetime.now() - pd.Timedelta(days=7))]
        st.metric("This Week", len(recent))
    
    # Score distribution chart
    if 'ats_score' in df and not df['ats_score'].isna().all():
        st.subheader("📈 ATS Score Distribution")
        fig = px.histogram(
            df, 
            x='ats_score', 
            nbins=20,
            title="Distribution of ATS Scores",
            labels={'ats_score': 'ATS Score', 'count': 'Number of Resumes'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline chart
    if len(df) > 1:
        st.subheader("📅 Analysis Timeline")
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        timeline_data = df.groupby(df['upload_date'].dt.date).size().reset_index()
        timeline_data.columns = ['Date', 'Count']
        
        fig = px.line(
            timeline_data,
            x='Date',
            y='Count',
            title="Resume Analyses Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent analyses table
    st.subheader("📋 Recent Analyses")
    display_df = df[['filename', 'upload_date', 'ats_score', 'layout_score']].head(10)
    if 'ats_score' in display_df:
        display_df['ats_score'] = display_df['ats_score'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    if 'layout_score' in display_df:
        display_df['layout_score'] = display_df['layout_score'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

def save_analysis_to_db(scores):
    """Save analysis results to database"""
    try:
        conn = sqlite3.connect('resume_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resume_analysis 
            (filename, upload_date, ats_score, layout_score, tfidf_score, sbert_score, missing_skills, analysis_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            "uploaded_resume.pdf",  # You might want to get the actual filename
            datetime.now(),
            scores.get('final_score', 0),
            scores.get('layout_score', 0),
            scores.get('tfidf_score', 0),
            scores.get('sbert_score', 0),
            json.dumps(scores.get('missing_skills', [])),
            json.dumps(scores)
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")

if __name__ == "__main__":
    main()