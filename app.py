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
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    return f'<a href="data:file/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'

def display_ats_score(score):
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
    init_db()
    init_session_state()
    
    st.title("**Multi-Granularity Multi-Modal Resume Analyzer**")
    st.markdown("Optimize your resume for ATS systems and job roles.")

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
        
        conn = sqlite3.connect('resume_data.db')
        total_resumes = pd.read_sql_query("SELECT COUNT(*) as count FROM resume_analysis", conn).iloc[0]['count']
        conn.close()
        
        st.metric("Total Resumes Analyzed", total_resumes)
    
    if page == "Resume Upload & Analysis":
        resume_upload_page()
    elif page == "Job Matching":
        job_matching_page()
    elif page == "Resume Optimization":
        resume_optimization_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()

def resume_upload_page():
    st.header("📄 Resume Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx'],
        help="Upload your resume in PDF or DOCX format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Parsing resume..."):
            parser = ResumeParser()
            parsed_data = parser.parse_resume(uploaded_file)
            st.session_state.parsed_resume = parsed_data
        
        with st.spinner("Analyzing resume..."):
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(parsed_data)
            st.session_state.analysis_results = analysis_results
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Extracted Information")
            if parsed_data.get('contact_info'):
                st.write("**Contact Information:**")
                for key, value in parsed_data['contact_info'].items():
                    if value:
                        st.write(f"- {key.title()}: {value}")
            
            if parsed_data.get('skills'):
                st.write("**Skills Extracted:**")
                st.write(", ".join(parsed_data['skills'][:10]))

        with col2:
            st.subheader("🔍 Structure Analysis")
            layout_score = analysis_results.get('layout_score', 0.5)
            st.metric("Layout Compliance", f"{layout_score:.1%}")
            
            if analysis_results.get('issues'):
                st.write("**Issues Found:**")
                for issue in analysis_results['issues']:
                    st.warning(f"⚠️ {issue}")
            
            if analysis_results.get('sections'):
                st.write("**Sections Detected:**")
                for section in analysis_results['sections']:
                    st.success(f"✅ {section}")
        
        st.subheader("📝 Parsed Text Preview")
        with st.expander("View extracted text"):
            st.text_area("Resume Text", parsed_data.get('text', ''), height=300, disabled=True)

def job_matching_page():
    st.header("🎯 Job Description Matching")
    
    if st.session_state.parsed_resume is None:
        st.warning("Please upload and analyze a resume first!")
        return
    
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        value=st.session_state.job_description,
        height=200
    )
    
    if st.button("🔍 Analyze Match", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description!")
            return
        
        st.session_state.job_description = job_description
        
        with st.spinner("Computing ATS scores..."):
            scorer = ATSScorer()
            scores = scorer.compute_ats_score(
                st.session_state.parsed_resume,
                job_description,
                st.session_state.analysis_results
            )
            st.session_state.ats_scores = scores
        
        display_ats_score(scores['final_score'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Semantic Similarity", f"{scores['sbert_score']:.1%}")
        with col2:
            st.metric("Keyword Match", f"{scores['tfidf_score']:.1%}")
        with col3:
            st.metric("Layout Score", f"{scores['layout_score']:.1%}")
        
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
        
        st.subheader("💡 Recommendations")
        for rec in scores.get('recommendations', []):
            st.info(f"💡 {rec}")
        
        save_analysis_to_db(scores)

def resume_optimization_page():
    st.header("✨ AI-Powered Resume Optimization")
    
    if st.session_state.parsed_resume is None or st.session_state.ats_scores is None:
        st.warning("Please analyze a resume and run job matching first!")
        return
    
    st.subheader("🤖 AI Resume Rewriter")
    
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
            optimized_resume = rewriter.rewrite_resume(
                st.session_state.parsed_resume,
                st.session_state.job_description,
                st.session_state.ats_scores,
                sections=rewrite_sections,
                style=template_style
            )
            
            pdf_path = rewriter.generate_pdf(optimized_resume, template_style)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 Original Resume")
                st.text_area("Original", st.session_state.parsed_resume.get('text', ''), height=400, disabled=True)
            with col2:
                st.subheader("✨ Optimized Resume")
                st.text_area("Optimized", optimized_resume.get('text', ''), height=400, disabled=True)
            
            if os.path.exists(pdf_path):
                st.success("✅ Optimized resume generated successfully!")
                st.markdown(create_download_link(pdf_path, "optimized_resume.pdf"), unsafe_allow_html=True)
            
            st.subheader("📈 Improvements Made")
            for improvement in optimized_resume.get('improvements', []):
                st.info(f"✨ {improvement}")

def analytics_dashboard_page():
    st.header("📊 Analytics Dashboard")
    
    conn = sqlite3.connect('resume_data.db')
    df = pd.read_sql_query("SELECT * FROM resume_analysis ORDER BY upload_date DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("No analysis data available yet.")
        return

    # 🔧 FIX — Convert upload_date from string to datetime
    df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resumes", len(df))
    with col2:
        st.metric("Average ATS Score", f"{df['ats_score'].mean():.1%}")
    with col3:
        st.metric("High-Quality Resumes", (df['ats_score'] >= 0.7).sum())
    with col4:
        recent = df[df['upload_date'] >= (datetime.now() - pd.Timedelta(days=7))]
        st.metric("This Week", len(recent))
    
    if 'ats_score' in df:
        st.subheader("📈 ATS Score Distribution")
        fig = px.histogram(df, x='ats_score', nbins=20, title="ATS Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(df) > 1:
        st.subheader("📅 Analysis Timeline")
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
        timeline = df.groupby(df['upload_date'].dt.date).size().reset_index(name='Count')
        
        fig = px.line(timeline, x='upload_date', y='Count', title="Resume Analyses Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Recent Analyses")
    display_df = df[['filename', 'upload_date', 'ats_score', 'layout_score']].head(10)
    display_df['ats_score'] = display_df['ats_score'].apply(lambda x: f"{x:.1%}")
    display_df['layout_score'] = display_df['layout_score'].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True)

def save_analysis_to_db(scores):
    try:
        conn = sqlite3.connect('resume_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resume_analysis 
            (filename, upload_date, ats_score, layout_score, tfidf_score, sbert_score, missing_skills, analysis_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            "uploaded_resume.pdf",
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
