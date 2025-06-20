from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavfile
from transformers import pipeline  # For STT (Whisper)
from kokoro import KPipeline  # For Kokoro-82M TTS
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize conversation memory
memory = ConversationBufferMemory()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b-latest",
    google_api_key=GEMINI_API_KEY
)

# Initialize Kokoro-82M TTS pipeline
tts_pipeline = KPipeline(lang_code='a')  # 'a' for American English

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
    template='''You are an interviewer agent. Evaluate the user's answer to: "{question}". Answer: "{answer}". History: {history}. Provide feedback with a score (0-100), comments, and suggestions. Return the response as valid JSON matching the schema: {{"score": integer, "comments": string, "suggestions": string}}. Use double quotes for strings, escape special characters (e.g., \\n, \\", \\*), and ensure the output is valid JSON without code fences, single quotes, or extra text.'''
)

prompt_question = PromptTemplate(
    input_variables=["role", "history"],
    template='''You are an interviewer agent. Generate one interview question for a {role}. History: {history}. Generate a question with its category (e.g., Technical, Behavioural) and difficulty (e.g., Easy, Medium, Hard). Return the response as valid JSON matching the schema: {{"question": string, "category": string, "difficulty": string}}. Use double quotes for strings, escape special characters (e.g., \\n, \\", \\*), and ensure the output is valid JSON without code fences, single quotes, or extra text.'''
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
    feedback = chain.invoke({"question": question, "answer": answer, "history": history})
    logging.info(f"Generated feedback: {feedback}")
    return feedback

# Function to generate interview question
def generate_question(role: str, history: str) -> InterviewQuestion:
    chain = prompt_question | llm.with_structured_output(InterviewQuestion)
    question = chain.invoke({"role": role, "history": history})
    logging.info(f"Generated question: {question}")
    return question

# Speech-to-text pipeline
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Record audio function
def record_audio(filename="input.wav", duration=5, sample_rate=16000):
    print("Recording audio")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wavfile.write(filename, sample_rate, audio)
    print("Recording saved as", filename)

# Text-to-speech function using Kokoro-82M
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
def test_interview_flow(max_question=3):
    print("Please say the role you want to interview for (e.g., Software Engineer).")
    record_audio()
    user_input = audio_to_text()
    print("Transcribed input:", user_input)
    role = get_interview_role(user_input)
    print("Selected role:", role.content)
    if not role.content.startswith("Could you clarify"):
        for i in range(max_question):
            print(f"\nQuestion {i+1} of {max_question}")
            history = memory.load_memory_variables({})['history']
            question = generate_question(role.content, history)
            print(f"Interviewer: {question.question} (Category: {question.category}, Difficulty: {question.difficulty})")
            text_audio(question.question, filename=f"question_{i+1}.wav")
            print("Please say your answer to the question")
            record_audio(duration=10, filename="answer.wav")
            answer = audio_to_text(filename="answer.wav")
            print("Your answer:", answer)
            if answer.lower() in ["stop", "end interview", "exit"]:
                print("Interviewer: Interview ended")
                break
            memory.save_context({"input": question.question}, {"output": answer})
            feedback = generate_feedback(question.question, answer, history)
            print(f"Feedback: {feedback.comments} (Score: {feedback.score})")
            print(f"Suggestions: {feedback.suggestions}")
            feedback_text = f"Feedback: {feedback.comments}. Score: {feedback.score}. Suggestions: {feedback.suggestions}"
            text_audio(feedback_text, filename=f"feedback_{i+1}.wav")
    else:
        print("Interviewer:", role.content)
        text_audio(role.content, filename="clarification.wav")

# Test STT function
def test_stt():
    record_audio()
    text = audio_to_text()
    print("You said:", text)

if __name__ == "__main__":
    test_interview_flow()