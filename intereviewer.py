from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavfile
from transformers import pipeline
from kokoro import KPipeline
import numpy as np
import logging
import nltk
from nltk.tokenize import sent_tokenize
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
import json

nltk.download('punkt_tab')
os.environ["ESPEAK_DATA_PATH"] = "C:\\Program Files\\eSpeak NG\\espeak-ng-data"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
print('GEMINI_API_KEY',GEMINI_API_KEY)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=GEMINI_API_KEY
)

# Initialize Kokoro-82M TTS pipeline
tts_pipeline = KPipeline(lang_code='a')

# Initialize embeddings for RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load questions from JSON file
def load_questions(file_path="questions.json"):
    with open(file_path, 'r') as f:
        questions_data = json.load(f)
    documents = [
        Document(
            page_content=q["question"],
            metadata={
                "role": q["role"],
                "category": q["category"],
                "difficulty": q["difficulty"]
            }
        ) for q in questions_data
    ]
    return documents

# Initialize FAISS vector store
def initialize_vector_store(documents):
    vector_store = FAISS.from_documents(documents, embeddings)
    print('shumu',documents)
    return vector_store

# Define Pydantic models
class Feedback(BaseModel):
    score: int = Field(description="Score out of 100")
    comments: str = Field(description="Constructive feedback")
    suggestions: str = Field(description="Suggestions for improvement")

class InterviewQuestion(BaseModel):
    question: str = Field(description="The interview question")
    category: str = Field(description="Category, e.g., Technical, Behavioural")
    difficulty: str = Field(description="Difficulty level, e.g., Easy, Medium, Hard")

# Define prompts
prompt_role = PromptTemplate(
    input_variables=["user_input"],
    template="You are an interviewer agent. The user said: '{user_input}'. Extract the job role they want to interview for (e.g., Software Engineer, Data Scientist). If unclear, ask: 'Could you clarify the role you want to interview for?' Respond with only the role name or the clarification question."
)

prompt_feedback = PromptTemplate(
    input_variables=["question", "answer", "history"],
    template='''You are an interviewer agent. Evaluate the user's answer to: "{question}". Answer: "{answer}". History: {history}. Provide feedback as valid JSON with:
    - "score": an integer (0-100) based on answer quality.
    - "comments": a string with constructive feedback and a humorous roast of the user's response.
    - "suggestions": a string with specific improvements and an example of how to answer the question better.
    Ensure the output is valid JSON with double quotes, escaped characters (e.g., \\n, \\", \\*), and no extra text.'''
)

prompt_question = PromptTemplate(
    input_variables=["role", "history", "retrieved_questions"],
    template='''You are an interviewer agent. Generate one interview question for a {role}. Ensure the question is diverse, not repetitive, and strictly pick up one question at a time from {retrieved_questions} on priority if it is not asked previously else you can choose to pickup best question according to {role} . History: {history}. Retrieved questions for inspiration: {retrieved_questions}. Return the response as valid JSON: {{"question": string, "category": string, "difficulty": string}}. Use double quotes, escape special characters, and ensure valid JSON.'''
)

# Function to extract interview role
def get_interview_role(user_input: str) -> str:
    chain = prompt_role | llm
    response = chain.invoke({"user_input": user_input})
    logging.info(f"Extracted role response: {response.content}")
    return response

# Function to generate feedback
def generate_feedback(question: str, answer: str, history: str) -> Feedback:
    chain = prompt_feedback | llm.with_structured_output(Feedback)
    full_response = ""
    current_sentence = ""
    for chunk in chain.stream({"question": question, "answer": answer, "history": history}):
        token = chunk.comments
        full_response += token
        current_sentence += token
        sentences = sent_tokenize(current_sentence)
        if len(sentences) > 1:
            sentence = sentences[0]
            current_sentence = "".join(sentences[1:])
            logging.info(f"Streaming sentence: {sentence}")
            text_audio(sentence)
    if current_sentence.strip():
        logging.info(f"Streaming final sentence: {current_sentence}")
        text_audio(current_sentence)
    return full_response

# Function to generate interview question with RAG
def generate_question(role: str, history: str, vector_store, user_preferences: dict = None) -> InterviewQuestion:
    # Apply user preferences (e.g., category, difficulty)
    query = f"Interview question for {role}"
    if user_preferences:
        if "category" in user_preferences:
            query += f" category: {user_preferences['category']}"
        if "difficulty" in user_preferences:
            query += f" difficulty: {user_preferences['difficulty']}"
    
    # Retrieve relevant questions
    retrieved_docs = vector_store.similarity_search(query, k=3)
    retrieved_questions = [doc.page_content for doc in retrieved_docs]
    retrieved_metadata = [doc.metadata for doc in retrieved_docs]
    
    # Format retrieved questions for the prompt
    retrieved_text = "; ".join([f"{q} (Category: {m['category']}, Difficulty: {m['difficulty']})" for q, m in zip(retrieved_questions, retrieved_metadata)])
    print('text',retrieved_text)
    
    # Generate question using LLM
    chain = prompt_question | llm.with_structured_output(InterviewQuestion)
    question = chain.invoke({"role": role, "history": history, "retrieved_questions": retrieved_text})
    
    # Ensure the question is not in history
    if question.question in history:
        logging.info("Question repeated, generating another...")
        return generate_question(role, history, vector_store, user_preferences)
    
    logging.info(f"Generated question: {question}")
    return question

# Speech-to-text pipeline
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", return_timestamps=True)

# Record audio function
def record_audio(filename="input.wav", duration=5, sample_rate=16000):
    print("Recording audio")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wavfile.write(filename, sample_rate, audio)
    print("Recording saved as", filename)

# Text-to-speech function
def text_audio(text: str, filename="output.wav", sample_rate=24000, voice='af_bella', speed=1.0):
    print("Generating audio for text...")
    generator = tts_pipeline(text, voice=voice, speed=speed)
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunks.append(audio)
    combined_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    sf.write(filename, combined_audio, sample_rate)
    print(f"Audio saved as {filename}")
    print("Playing audio...")
    audio_data, _ = sf.read(filename)
    sd.play(audio_data, sample_rate)
    sd.wait()

# Audio-to-text function
def audio_to_text(filename="input.wav"):
    result = stt_pipeline(filename)
    return result["text"]

# Test interview flow
def test_interview_flow(max_questions=1, user_preferences=None):
    # Initialize vector store
    documents = load_questions("questions.json")
    vector_store = initialize_vector_store(documents)
    
    print("Please say the role you want to interview for (e.g., Software Engineer).")
    record_audio()
    user_input = audio_to_text()
    print("Transcribed input:", user_input)
    role = get_interview_role(user_input)
    print("Selected role:", role.content)
    
    if not role.content.startswith("Could you clarify"):
        for i in range(max_questions):
            print(f"\nQuestion {i+1} of {max_questions}")
            history = memory.load_memory_variables({})['history']
            question = generate_question(role.content, history, vector_store, user_preferences)
            print(f"Interviewer: {question.question} (Category: {question.category}, Difficulty: {question.difficulty})")
            text_audio(question.question, filename=f"question_{i+1}.wav")
            print("Please say your answer to the question")
            record_audio(duration=40, filename="answer.wav")
            answer = audio_to_text(filename="answer.wav")
            print("Your answer:", answer)
            if answer.lower() in ["stop", "end interview", "exit"]:
                print("Interviewer: Interview ended")
                break
            memory.save_context({"input": question.question}, {"output": answer})
            feedback = generate_feedback(question.question, answer, history)
    else:
        print("Interviewer:", role.content)
        text_audio(role.content, filename="clarification.wav")

if __name__ == "__main__":
    # Example user preferences
    user_preferences = {"category": "Technical", "difficulty": "Medium"}
    test_interview_flow(max_questions=10, user_preferences=user_preferences)