import os
import tempfile
import threading
import logging
import uuid

import gradio as gr
import pygame
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document


# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
API_KEY = os.getenv("API_KEY")

# 로그 설정
logging.basicConfig(
    filename="langchain_voice_assistant_logs.txt",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# LangChain Memory 및 RAG 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
# # 문서 리스트 추가
# documents = [
#     Document(page_content="LangChain은 AI 개발을 위한 도구입니다.", metadata={"source": "doc1"}),
#     Document(page_content="FAISS는 효율적인 벡터 검색을 지원합니다.", metadata={"source": "doc2"}),
# ]
# document_store = FAISS.from_documents(documents, embeddings)
# retriever = document_store.as_retriever()

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["history", "question"],
    template="""
    너는 사용자의 음성 비서야. 다음은 대화의 기록이고, 사용자가 추가 질문을 했다:
    기록: {history}
    질문: {question}
    대답을 제공해라.
    """
)

# LangChain Conversational Retrieval Chain 초기화
chain = ConversationalRetrievalChain(
    # retriever=retriever,
    memory=memory,
    llm=OpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=API_KEY
    ),
    prompt=prompt_template,
)

# 사용자별 임시 디렉토리 생성
def get_user_temp_dir():
    user_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"langchain_assistant_{user_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# STT 모듈 호출 (Whisper API 사용)
def speech_to_text(file_path):
    logging.info(f"STT 요청: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            response = chain.llm.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return response.strip()
    except Exception as e:
        logging.error(f"STT 오류: {e}")
        return "음성 인식 중 오류가 발생했습니다."

# TTS 모듈 호출 (OpenAI TTS 사용)
def speak_async(text, user_temp_dir):
    def run_tts():
        try:
            output_file = os.path.join(user_temp_dir, "output.mp3")
            response = chain.llm.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            response.write_to_file(output_file)

            pygame.mixer.init()
            pygame.mixer.music.load(output_file, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            pygame.mixer.quit()
        except Exception as e:
            logging.error(f"TTS 오류: {e}")

    tts_thread = threading.Thread(target=run_tts)
    tts_thread.daemon = True
    tts_thread.start()

# 음성 파일 처리 및 응답 생성
def process_audio(mic):
    if not mic:
        return "오디오 파일이 없습니다."

    user_temp_dir = get_user_temp_dir()

    try:
        # STT 단계
        command = speech_to_text(mic)

        # ChatGPT 및 RAG 단계
        response = chain.run({"history": memory.load_memory(), "question": command})

        # TTS 단계
        speak_async(response, user_temp_dir)

        return f"인식된 텍스트: {command}\nLangChain 응답: {response}"
    except Exception as e:
        logging.error(f"오류: {e}")
        return "오디오 처리 중 오류가 발생했습니다."

# Gradio 인터페이스
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=['microphone'], type="filepath", format="mp3", editable=False, render=False),
    outputs="text",
    title="LangChain 기반 AI 음성 비서",
    description="STT -> ChatGPT -> TTS의 작업 순서로 진행되며, 메모리 및 정보 검색 기능이 포함됩니다.",
    live=True,
)

if __name__ == "__main__":
    interface.launch(share=True)
