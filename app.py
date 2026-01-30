"""
XplainAI - Explainable AI Platform
Making Machine Learning Transparent and Understandable
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ml_models import (
    MLModelManager, 
    get_available_datasets, 
    get_available_algorithms,
    get_algorithm_descriptions
)

# Page configuration
st.set_page_config(
    page_title="XplainAI - Explainable Machine Learning",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cute and simple theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #FF6B9D;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --bg-light: #F8F9FA;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Step indicators */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 0;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        margin: 0 0.5rem;
        border-radius: 10px;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    .step.completed {
        background: #4CAF50;
        color: white;
    }
    
    .step-number {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .step-title {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Result cards */
    .result-approved {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(76,175,80,0.3);
    }
    
    .result-rejected {
        background: linear-gradient(135deg, #F44336 0%, #e53935 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(244,67,54,0.3);
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102,126,234,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102,126,234,0.4);
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
    }
    
    /* Explanation box */
    .explanation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: #e0e0e0;
        border-radius: 8px;
        height: 30px;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .feature-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding: 0 1rem;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .step-indicator {
            flex-direction: column;
        }
        
        .step {
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'algorithm' not in st.session_state:
        st.session_state.algorithm = None
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'input_data' not in st.session_state:
        st.session_state.input_data = None


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🔍 XplainAI</h1>
        <p>Making Machine Learning Decisions Transparent and Understandable</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_indicator():
    """Render step indicator"""
    steps = [
        ("1", "Choose Dataset"),
        ("2", "Select Algorithm"),
        ("3", "Enter Details"),
        ("4", "Get Result")
    ]
    
    step_html = '<div class="step-indicator">'
    for i, (num, title) in enumerate(steps, 1):
        if i < st.session_state.step:
            step_class = "step completed"
        elif i == st.session_state.step:
            step_class = "step active"
        else:
            step_class = "step"
        
        step_html += f'''
        <div class="{step_class}">
            <div class="step-number">{num}</div>
            <div class="step-title">{title}</div>
        </div>
        '''
    
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)


def step1_select_dataset():
    """Step 1: Dataset Selection"""
    st.markdown("### 📊 Step 1: Choose Your Problem Type")
    st.markdown("Select a real-world scenario to explore how AI makes decisions:")
    
    datasets = get_available_datasets()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🏦 Loan Approval</h3>
            <p><strong>Use Case:</strong> Banking & Finance</p>
            <p>Understand why a loan application is approved or rejected based on credit score, income, employment history, and more.</p>
            <p><strong>Target Users:</strong> Loan applicants, banks, financial institutions</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Loan Approval", key="loan_btn", use_container_width=True):
            st.session_state.dataset = "Loan Approval"
            st.session_state.model_manager = MLModelManager("Loan Approval")
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎓 Student Admission</h3>
            <p><strong>Use Case:</strong> Education</p>
            <p>Learn why a student is admitted or denied based on GPA, test scores, extracurricular activities, and essays.</p>
            <p><strong>Target Users:</strong> Students, universities, educators</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Student Admission", key="student_btn", use_container_width=True):
            st.session_state.dataset = "Student Admission"
            st.session_state.model_manager = MLModelManager("Student Admission")
            st.session_state.step = 2
            st.rerun()


def step2_select_algorithm():
    """Step 2: Algorithm Selection"""
    st.markdown("### 🤖 Step 2: Choose Your AI Algorithm")
    st.markdown("Each algorithm has different strengths. Pick one to see how it works:")
    
    algorithms = get_available_algorithms()
    descriptions = get_algorithm_descriptions()
    
    # Train models if not already trained
    if not st.session_state.model_manager.is_trained:
        with st.spinner("Training models... This will only take a moment! 🚀"):
            st.session_state.model_manager.train_models()
    
    cols = st.columns(2)
    
    for i, algo in enumerate(algorithms):
        with cols[i % 2]:
            accuracy = st.session_state.model_manager.models[algo]['accuracy']
            
            st.markdown(f"""
            <div class="info-card">
                <h3>{algo}</h3>
                <p>{descriptions[algo]}</p>
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{accuracy*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Use {algo}", key=f"algo_{i}", use_container_width=True):
                st.session_state.algorithm = algo
                st.session_state.step = 3
                st.rerun()
    
    if st.button("← Back", key="back_from_algo"):
        st.session_state.step = 1
        st.rerun()


def step3_enter_details():
    """Step 3: Enter Input Details"""
    st.markdown("### 📝 Step 3: Enter Your Details")
    st.markdown("Fill in the information below. Feel free to experiment with different values!")
    
    dataset = st.session_state.model_manager.dataset
    feature_names = dataset.get_feature_names()
    feature_descriptions = dataset.get_feature_descriptions()
    input_ranges = dataset.get_input_ranges()
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    
    input_data = []
    
    # Create input fields based on dataset
    if st.session_state.dataset == "Loan Approval":
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input(
                "💳 Credit Score",
                min_value=int(input_ranges['credit_score'][0]),
                max_value=int(input_ranges['credit_score'][1]),
                value=int(input_ranges['credit_score'][2]),
                step=10,
                help="Your credit score (300-850). Higher is better!"
            )
            
            annual_income = st.number_input(
                "💰 Annual Income ($)",
                min_value=int(input_ranges['annual_income'][0]),
                max_value=int(input_ranges['annual_income'][1]),
                value=int(input_ranges['annual_income'][2]),
                step=5000,
                help="Your total annual income"
            )
            
            loan_amount = st.number_input(
                "🏠 Loan Amount ($)",
                min_value=int(input_ranges['loan_amount'][0]),
                max_value=int(input_ranges['loan_amount'][1]),
                value=int(input_ranges['loan_amount'][2]),
                step=10000,
                help="Amount you want to borrow"
            )
        
        with col2:
            employment_years = st.number_input(
                "💼 Years of Employment",
                min_value=float(input_ranges['employment_years'][0]),
                max_value=float(input_ranges['employment_years'][1]),
                value=float(input_ranges['employment_years'][2]),
                step=0.5,
                help="How long you've been employed"
            )
            
            debt_to_income = st.number_input(
                "📊 Debt-to-Income Ratio",
                min_value=float(input_ranges['debt_to_income_ratio'][0]),
                max_value=float(input_ranges['debt_to_income_ratio'][1]),
                value=float(input_ranges['debt_to_income_ratio'][2]),
                step=0.05,
                format="%.2f",
                help="Your monthly debt payments divided by income (lower is better)"
            )
        
        input_data = [credit_score, annual_income, loan_amount, employment_years, debt_to_income]
    
    elif st.session_state.dataset == "Student Admission":
        col1, col2 = st.columns(2)
        
        with col1:
            gpa = st.number_input(
                "📚 GPA (Grade Point Average)",
                min_value=float(input_ranges['gpa'][0]),
                max_value=float(input_ranges['gpa'][1]),
                value=float(input_ranges['gpa'][2]),
                step=0.1,
                format="%.2f",
                help="Your GPA on a 4.0 scale"
            )
            
            test_score = st.number_input(
                "✏️ Test Score",
                min_value=int(input_ranges['test_score'][0]),
                max_value=int(input_ranges['test_score'][1]),
                value=int(input_ranges['test_score'][2]),
                step=50,
                help="Your standardized test score (e.g., SAT)"
            )
        
        with col2:
            extracurricular = st.number_input(
                "🎭 Extracurricular Activities",
                min_value=int(input_ranges['extracurricular_count'][0]),
                max_value=int(input_ranges['extracurricular_count'][1]),
                value=int(input_ranges['extracurricular_count'][2]),
                step=1,
                help="Number of clubs, sports, or activities"
            )
            
            essay_score = st.number_input(
                "📄 Essay Score",
                min_value=float(input_ranges['essay_score'][0]),
                max_value=float(input_ranges['essay_score'][1]),
                value=float(input_ranges['essay_score'][2]),
                step=0.5,
                format="%.1f",
                help="Quality of your application essay (1-10)"
            )
        
        input_data = [gpa, test_score, extracurricular, essay_score]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back", key="back_from_input"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("🔮 Get Prediction", key="predict_btn", use_container_width=True):
            st.session_state.input_data = input_data
            
            # Make prediction
            with st.spinner("Analyzing your data... 🤔"):
                prediction, confidence = st.session_state.model_manager.predict(
                    st.session_state.algorithm,
                    input_data
                )
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.step = 4
            st.rerun()


def step4_show_results():
    """Step 4: Show Results and Explanation"""
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence
    
    # Display result
    if prediction == 1:
        st.markdown("""
        <div class="result-approved fade-in">
            <div class="result-icon">✅</div>
            <div class="result-title">APPROVED!</div>
            <div class="result-subtitle">Congratulations! Your application has been approved.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-rejected fade-in">
            <div class="result-icon">❌</div>
            <div class="result-title">NOT APPROVED</div>
            <div class="result-subtitle">Unfortunately, your application was not approved at this time.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show confidence if available
    if confidence is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Get explanation
    st.markdown("### 🔍 Why This Decision?")
    st.markdown("Here's a clear explanation of the factors that influenced this decision:")
    
    explanation = st.session_state.model_manager.explain_decision(
        st.session_state.algorithm,
        st.session_state.input_data,
        prediction
    )
    
    st.markdown(f"""
    <div class="explanation-box fade-in">
        {explanation.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance visualization
    st.markdown("### 📊 Feature Importance")
    st.markdown("See which factors had the biggest impact on the decision:")
    
    importance = st.session_state.model_manager.get_feature_importance(st.session_state.algorithm)
    
    if importance:
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        feature_descriptions = st.session_state.model_manager.dataset.get_feature_descriptions()
        
        for feature, imp in sorted_importance:
            display_name = feature_descriptions.get(feature, feature)
            percentage = imp * 100
            
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span><strong>{display_name}</strong></span>
                    <span>{percentage:.1f}%</span>
                </div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: {percentage}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualization using plotly
    if importance:
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=[feature_descriptions.get(k, k) for k in importance.keys()],
            orientation='h',
            marker=dict(
                color=list(importance.values()),
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{v*100:.1f}%' for v in importance.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Feature Importance Breakdown",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Try Again", key="try_again", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("🔀 Change Algorithm", key="change_algo", use_container_width=True):
            st.session_state.step = 2
            st.session_state.prediction = None
            st.rerun()
    
    with col3:
        if st.button("🏠 Start Over", key="start_over", use_container_width=True):
            st.session_state.step = 1
            st.session_state.dataset = None
            st.session_state.algorithm = None
            st.session_state.prediction = None
            st.rerun()


def render_sidebar():
    """Render sidebar with info"""
    with st.sidebar:
        st.markdown("## 🎯 About XplainAI")
        st.markdown("""
        XplainAI makes machine learning decisions transparent and easy to understand.
        
        ### 🌟 Features
        - **Multiple Datasets**: Real-world scenarios
        - **Various Algorithms**: Compare different AI approaches
        - **Instant Predictions**: Get results immediately
        - **Clear Explanations**: Understand the 'why'
        - **Visual Insights**: See what matters most
        
        ### 📚 How It Works
        1. Choose a problem type (dataset)
        2. Select an AI algorithm
        3. Enter your information
        4. Get instant prediction + explanation
        5. Learn and experiment!
        
        ### 🎓 Perfect For
        - Students learning ML
        - Educators teaching AI
        - Anyone curious about AI decisions
        - Building trust in AI systems
        """)
        
        if st.session_state.dataset:
            st.markdown("---")
            st.markdown(f"**Current Dataset:** {st.session_state.dataset}")
        
        if st.session_state.algorithm:
            st.markdown(f"**Current Algorithm:** {st.session_state.algorithm}")


def main():
    """Main application logic"""
    initialize_session_state()
    render_header()
    render_sidebar()
    render_step_indicator()
    
    # Route to appropriate step
    if st.session_state.step == 1:
        step1_select_dataset()
    elif st.session_state.step == 2:
        step2_select_algorithm()
    elif st.session_state.step == 3:
        step3_enter_details()
    elif st.session_state.step == 4:
        step4_show_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Made with ❤️ by XplainAI | Empowering Transparent AI</p>
        <p style="font-size: 0.9rem;">Built with Streamlit, scikit-learn, and Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
