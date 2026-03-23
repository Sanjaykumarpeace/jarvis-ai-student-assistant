import streamlit as st
import requests
from pypdf import PdfReader
import whisper
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import tempfile

def speak_text(text):

    tts = gTTS(text)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

model = whisper.load_model("base")

# Page configuration
st.set_page_config(
    page_title="JARVIS STUDENT ASSISTANT",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 JARVIS STUDENT ASSISTANT")
st.write("Your AI-powered study helper running locally with LLaMA3")

# Function to communicate with Ollama
def ask_ai(prompt):

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error connecting to Ollama. Make sure it is running."


# Feature selection
option = st.selectbox(
    "Choose a feature",
    [
        "Study Plan Generator",
        "Quiz Generator from Notes",
        "Research Paper Explainer",
        "Ask Questions From PDF",
        "Voice AI Assistant"
    ]
)

===============================

if option == "Study Plan Generator":

    st.header("📚 AI Study Plan Generator")

    subject = st.text_input("Enter subject")
    topics = st.text_area("Enter topics to study")
    days = st.number_input("Days until exam", min_value=1)

    if st.button("Generate Study Plan"):

        prompt = f"""
        Create a clear day-by-day study plan.

        Subject: {subject}
        Topics: {topics}
        Exam in {days} days.

        Make it structured and easy to follow.
        """
        with st.spinner("JARVIS IS THINKING..."):
            result = ask_ai(prompt)
        st.subheader("Your AI Study Plan")
        st.write(result)


===============================

elif option == "Quiz Generator from Notes":

    st.header("📝 Generate Quiz from Notes")

    notes = st.text_area("Paste your notes here")

    if st.button("Generate Quiz"):

        prompt = f"""
        Create 5 multiple choice questions from these notes.

        Provide answers as well.

        Notes:
        {notes}
        """
        with st.spinner("JARVIS IS THINKING..."):
            result = ask_ai(prompt)
        st.subheader("Generated Quiz")
        st.write(result)


===============================

elif option == "Research Paper Explainer":

    st.header("🔬 Research Paper Simplifier")

    paper = st.text_area("Paste research paper abstract or content")

    if st.button("Explain Research Paper"):

        prompt = f"""
        Explain this research paper in simple language.

        Also include:
        - Key ideas
        - Why it matters
        - Possible future research

        Paper:
        {paper}
        """
        with st.spinner("JARVIS IS THINKING..."):
            result = ask_ai(prompt)
        st.subheader("Simplified Explanation")
        st.write(result)

==============================

elif option == "Ask Questions From PDF":

    st.header("📄 Ask Questions From Your PDF")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        question = st.text_input("Ask a question about the document")

        if st.button("Get Answer"):

            prompt = f"""
            Use the following document to answer the question.

            Document:
            {text}

            Question:
            {question}
            """
            with st.spinner("JARVIS is analyzing the document..."):
                result = ask_ai(prompt)

            st.subheader("Answer")
            st.write(result)
elif option == "Voice AI Assistant":

    st.header("🎤 Voice AI Assistant")

    st.write("Record your question using the microphone.")

    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

    if audio:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            temp_audio_path = f.name

        st.audio(audio["bytes"], format="audio/wav")

        st.write("Transcribing audio...")

        result = model.transcribe(temp_audio_path)

        user_text = result["text"]

        st.write("You said:", user_text)

        prompt = f"""
        Answer this student question clearly:

        {user_text}
        """

        with st.spinner("JARVIS is thinking..."):
            answer = ask_ai(prompt)

        st.subheader("AI Response")
        st.write(answer)

        audio_file = speak_text(answer)

        st.audio(audio_file, format="audio/mp3")       

st.write("---")
st.write("⚡ Powered by Local AI using Ollama + LLaMA3")
