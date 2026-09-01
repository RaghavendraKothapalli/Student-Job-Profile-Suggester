import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Student Job Profile Suggester",
    page_icon="🎯",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2a2a2a;
        color: #9ca3af;
        font-size: 0.85rem;
        text-align: center;
    }
    .footer a {
        color: #38c9a0;
        text-decoration: none;
    }
    div.stButton > button {
        width: 100%;
        font-weight: 700;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model_path = "Models/best_modelsV2.joblib"

best_models, scaler, skill_encoder, profile_encoder = None, None, None, None
try:
    best_models, scaler, skill_encoder, profile_encoder = joblib.load(model_path)
except FileNotFoundError:
    st.error("Error: Model V2 file not found!")
    st.stop()


def get_user_input():
    skillsList = ('Angular', 'Ansible', 'BASH/SHELL', 'C/C++', 'Cisco Packet Tracer', 'Deep Learning',
                'Figma', 'GitHub', 'HTML/CSS', 'Java', 'JavaScript', 'Linux', 'Machine Learning', 'MySQL',
                'Node.js', 'Oracle', 'Photoshop', 'PyTorch', 'Python', 'R', 'React', 'TensorFlow', 'Wireshark')

    col1, col2 = st.columns(2)
    with col1:
        skill_1 = st.selectbox('Top skill #1', skillsList, index=21)
    with col2:
        skill_2 = st.selectbox('Top skill #2', skillsList, index=19)

    st.markdown("#### Academic scores")
    c1, c2 = st.columns(2)
    with c1:
        dsa = st.slider('DSA score (0-100)', min_value=0, max_value=100, value=72)
        dbms = st.slider('DBMS score (0-100)', min_value=0, max_value=100, value=74)
        os_score = st.slider('Operating Systems score (0-100)', min_value=0, max_value=100, value=73)
        cn = st.slider('Computer Networks score (0-100)', min_value=0, max_value=100, value=60)
        mathematics = st.slider('Mathematics score (0-100)', min_value=0, max_value=100, value=87)
    with c2:
        aptitude = st.slider('Aptitude score (0-100)', min_value=0, max_value=100, value=82)
        communication = st.slider('Communication score (0-100)', min_value=0, max_value=100, value=64)
        problem_solving = st.slider('Problem Solving score (0-10)', min_value=0, max_value=10, value=7)
        creativity = st.slider('Creativity score (0-10)', min_value=0, max_value=10, value=6)
        hackathons = st.slider('Number of Hackathons', min_value=0, max_value=10, value=4)

    user_skills = skill_encoder.transform([[skill_1, skill_2]]).toarray()

    user_input = [dsa, dbms, os_score, cn, mathematics, aptitude, communication, problem_solving, creativity, hackathons]

    numerical_features = scaler.transform(np.array(user_input).reshape(1, -1))
    user_input_transformed = np.hstack((numerical_features, user_skills))

    return user_input_transformed


def main():
    st.markdown('<div class="main-header">🎯 Student Job Profile Suggester</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Predicts your best-fit job profile from academic scores and technical '
        'skills, using an XGBoost model trained and tuned on student placement data.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to use this app", expanded=False):
        st.write(
            """
            1. Select your top two technical skills from the dropdowns.
            2. Use the sliders to enter your scores across core CS subjects and soft skills.
            3. Click **Predict Job Profile** to get your suggested role.
            """
        )

    st.warning("The prediction combines your scores and skills together — it isn't based on either alone.")

    user_input_original = get_user_input()

    if st.button('🔮 Predict Job Profile'):
        user_input = user_input_original.copy()
        predicted_profile = best_models['XGB'].predict(user_input)
        predicted_job = profile_encoder.inverse_transform(predicted_profile)
        st.success(f'**Predicted Job Profile: {predicted_job[0]}**')

    st.markdown(
        """
        <div class="footer">
        Built by <b>Raghavendra Kothapalli</b> — B.Tech CS (AI/ML), Lovely Professional University<br>
        <a href="https://github.com/RaghavendraKothapalli/Student-Job-Profile-Suggester" target="_blank">Source code on GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/raghavendrakothapalli/" target="_blank">Connect on LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
