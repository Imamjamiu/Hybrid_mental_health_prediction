
import streamlit as st
import sys
import os
import time

# Configure Streamlit first
st.set_page_config(
    page_title="Hybrid Mental Health Prediction System",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Path setup
current_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_folder)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

# Import project modules
try:
    from classifier import MentalHealthClassifier
    from recommender import MentalHealthRecommender

except ImportError as e:
    st.error(
        f"Critical Error: Could not import scripts from {src_path}.\n"
        f"Detail: {e}"
    )
    st.stop()

# configuration
MODEL_PATH = os.path.join(project_root, "models", "final_unified_model")



# CSS styling
st.markdown ("""
    <style>
    div.block-container {
        padding-top: 3rem; /* enough space for the arrow */
        padding-bottom: 2rem;
    }

    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    .high-risk { background-color: #ffebee; border: 1px solid #ffcdd2; color: #b71c1c; }
    .med-risk { background-color: #fff3e0; border: 1px solid #ffe0b2; color: #e65100; }
    .low-risk { background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #1b5e20; }

    div.stButton > button {
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_models():
    csv_path = os.path.join (src_path,"resources.csv")
    if not os.path.exists (csv_path):
        st.error (f" CRITICAL ERROR: Could not find 'resources.csv' at: {csv_path}")
        st.stop ()

    doctor = MentalHealthClassifier (MODEL_PATH)
    librarian = MentalHealthRecommender (MODEL_PATH,resource_file=csv_path)
    return doctor,librarian


with st.spinner ('🚀 Initializing Neural Networks...'):
    try:
        doctor,librarian = load_models ()
    except Exception as e:
        st.error (f"Error loading models: {e}")
        st.stop ()

# Sidebar (The Input Zone)
with st.sidebar:
    st.title ("👤 Patient Profile")
    st.markdown ("---")

    st.subheader("1. Behavioral Metrics")
    sleep = st.slider("😴 Avg Sleep (Hours)",0,12,6)
    cgpa = st.number_input("📚 Current CGPA", 0.0, 5.0, 3.5, step=0.05)

    st.markdown("---")
    st.subheader("2. Risk Factors")
    anxiety_status = st.radio ("Do you currently feel anxious?",["No","Yes"],horizontal=True)

    self_harm_check = st.radio(
        "Have you recently had thoughts of hurting yourself?",
        ["No", "Yes"],
        horizontal=True
    )

    st.info("ℹ️ Data is processed locally.")

# Main content area
st.markdown("""
    <div style="margin-bottom: 15px; font-size: 10px; color: #555; display: flex; align-items: center;">
        <span style="font-size: 11px; margin-right: 1px;">↖️</span>
        <span>Set your <b>Behavioral Metrics</b> first.</span>
    </div>
    """, unsafe_allow_html=True)


st.title("🧠 Hybrid Mental Health Prediction System")
st.caption("A Hybrid Mental Health Support System powered by RoBERTa")

st.subheader("💭 Share Your Thoughts")
journal_text = st.text_area(
    "How are you feeling right now?",
    height=200,
    placeholder="I feel overwhelmed by schoolwork and haven't slept well in days...(Mentioning things like academic pressure, sleep quality, emotions, or anxiety helps me analyze your mental health risks accurately)."
)

# button(centerd)
st.markdown("<br>", unsafe_allow_html=True) # Spacer
col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
with col_btn:
    analyze_btn = st.button("🔍 Analyze Mental Health", type="primary", use_container_width=True)

# Prediction logic
if analyze_btn:
    # Gibberish Filter
    if not journal_text or journal_text.count(" ") < 2:
        st.warning("⚠️ Please write a complete sentence (at least 3 words) to get an accurate prediction.")
    else:
        # Get Base Prediction from AI
        with st.spinner("Analyzing..."):
            time.sleep(1)
            prediction, risk_score = doctor.predict(journal_text)

        # Keep the model prediction and probability unchanged.
        logic_notes = []

        # Safety response for self-harm disclosure
        override_triggered = self_harm_check == "Yes"

        # result display
        tab1,tab2,tab3 = st.tabs(["📊 Analysis Report","💊 Recommended Resources","⚙️ Debug Info"])

        with tab1:
            st.markdown ("### Diagnosis Summary")

            # Safety notice for self-harm disclosure
            if override_triggered:
                st.error(
                    "Thank you for sharing this. You deserve support, and you do not "
                    "have to handle this alone. If you feel you may act on these "
                    "thoughts or cannot keep yourself safe, please go to the nearest "
                    "hospital emergency department or contact local emergency services. "
                    "If possible, reach out to someone you trust and ask them to stay "
                    "with you."
                )

            # Model Output
            st.markdown("### Model Output")

            st.info(
                "This is an experimental AI model prediction. "
                "Its score is not a clinical risk assessment or diagnosis."
            )

            col1,col2,col3 = st.columns ([1.3,1,1])

            with col1:
                st.markdown ('<p style="font-size: 14px; margin-bottom: 0;">Predicted Status</p>',
                             unsafe_allow_html=True)
                # This HTML allows the text to wrap to a second line if needed
                st.markdown (f'<h3 style="margin-top: 0;">{prediction}</h3>',unsafe_allow_html=True)

            col2.metric("Class 1 Model Score", f"{risk_score:.1%}")
            col3.metric("Sleep Balance",f"{sleep} hrs",delta=f"{sleep - 8} hrs" if sleep < 8 else "Good")

            st.caption("Class 1 model score (experimental, not clinically calibrated)")
            st.progress(risk_score)

            # Show logic notes if inputs changed the score
            if logic_notes:
                st.info (f"ℹ️ **Score Adjustment:** Risk increased due to: {', '.join (logic_notes)}")

        with tab2:
            st.markdown ("### 📚 Curated Interventions")
            recs = librarian.get_recommendations(
                journal_text,
                self_harm_disclosed=override_triggered
            )

            if override_triggered:
                st.warning(
                    "No verified local crisis contacts are configured in this app yet. "
                    "If you may act on these thoughts or cannot stay safe, please go "
                    "to the nearest hospital emergency department. Reach out to someone "
                    "you trust and ask them to stay with you."
                )
            elif not recs:
                st.info("No specific resources found for this context.")
            else:
                for r in recs:
                    color = "red" if r['Type'] == 'Hotline' else "blue"
                    icon = "☎️" if r['Type'] == 'Hotline' else "📖"
                    with st.expander (f"{icon} {r['Description']}"):
                        st.markdown (f"**Type:** :{color}[{r['Type']}]")
                        st.link_button (f"🔗 Open {r['Type']}",r['Link'])

        with tab3:
            st.json ({
                "model": "RoBERTa-FineTuned",
                "class_1_model_score": risk_score,
                "logic_adjustments": logic_notes,
                "self_harm_disclosed": override_triggered
            })

# footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.8em;'>
        ⚠️ <b>DISCLAIMER:</b> This is an AI-powered prototype for educational purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        If you are in crisis, please call emergency services immediately.
    </div>
    """,
    unsafe_allow_html=True
)
