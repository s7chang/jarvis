import os
import tempfile
import threading
import json
import logging
import uuid

import whisper
import gradio as gr
import pygame
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

ENV_STT_TYPE = os.getenv("STT_TYPE", "API").lower()
ENV_STT_MODEL = os.getenv("STT_MODEL").lower()
client = OpenAI(api_key=os.getenv("API_KEY"))

# 로그 설정
logging.basicConfig(
    filename="jarvis_logs.txt",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if ENV_STT_TYPE == "local":
    model = whisper.load_model(ENV_STT_MODEL)

# 사용자별 임시 디렉토리 생성 함수
def get_user_temp_dir():
    user_id = str(uuid.uuid4())  # 고유 사용자 ID 생성
    temp_dir = os.path.join(tempfile.gettempdir(), f"jarvis_{user_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# ChatGPT API를 호출하는 함수
def chatgpt_response(prompt):
    logging.info(f"ChatGPT 요청: {prompt}")

    content = f"""
    너는 음성 AI 비서야. 너의 역할은 사용자의 요청을 분석하고 요청의 카테고리를 정한 뒤 응답을 작성하는 것이야. 

    ### 네가 처리할 수 있는 카테고리:
    - 일정관리: 알람 설정, 메일 연동, 일정 추가/변경/삭제
    - 명함관리: 명함 저장 및 검색
    - 메모: 메모 저장 및 검색
    - 메일 보내기: 메일 작성 및 전송
    - 문자 보내기: 문자 작성 및 전송
    - 채팅 시 멘트 추천
    - 전화하기: 전화번호로 전화 연결
    - 음악듣기: 음악 재생 요청

    ### 응답 형식:
    1. 요청이 위의 카테고리에 해당할 경우:
    - {{"category": "카테고리 이름", "content": [실제 동작 시 필요한 내용]}}

    2. 요청이 위의 카테고리에 해당하지 않을 경우:
    - {{"category": "응답", "content": "[요청의 내용에 대한 응답]"}}

    ### 사용자의 요청:
    {prompt}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}]
        )
        response_text = response.choices[0].message.content.strip("'''").strip('"""').strip('```').strip('json').strip('-').strip()
        logging.info(f"ChatGPT 응답: {response_text}")
        return response_text
    except Exception as e:
        logging.error(f"ChatGPT 호출 중 오류 발생: {e}")
        return f"ChatGPT 호출 중 오류가 발생했습니다: {e}"

# 음성 출력 비동기 처리
def speak_async(text, user_temp_dir):
    def run_tts():
        try:
            output_file = os.path.join(user_temp_dir, "output.mp3")
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            response.write_to_file(output_file)
            # pygame으로 음성 재생
            pygame.mixer.init()
            pygame.mixer.music.load(output_file, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            pygame.mixer.quit()
        except Exception as e:
            logging.error(f"TTS 오류: {e}")

    tts_thread = threading.Thread(target=run_tts)
    tts_thread.daemon = True  # 메인 프로그램 종료 시 강제 종료
    tts_thread.start()

# 응답을 음성으로 변환하는 함수
def generate_audio_output(response, user_temp_dir):
    try:
        response_data = json.loads(response)
        category = response_data.get("category", "응답")
        content = response_data.get("content", "요청 내용을 처리할 수 없습니다.")
        
        # 카테고리별 음성 출력
        if category == "응답":
            speech_text = content
        else:
            speech_text = f"{category} 작업을 시작합니다. {content}"

        speak_async(speech_text, user_temp_dir)
    except Exception as e:
        speak_async("응답이 잘못된 JSON 형태로 전달되었습니다.", user_temp_dir)
        logging.error(f"generate_audio_output 오류: {e}, {response}")


# whisper api 사용
def speech_to_text(file):
    logging.info(f"STT_TYPE: {ENV_STT_TYPE}, file: {file}")
    try:
        if model:
            result = model.transcribe(file)
            response = result["text"]
        else:
            # OpenAI Whisper API 호출
            with open(file, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"  # 텍스트 형식의 응답 요청
                )
    except Exception as e:
        logging.error(f"STT_TYPE: {ENV_STT_TYPE}, speech_to_text_by_api() error : {e}")

    return response

# 음성 파일 처리 및 응답 생성 함수
def korean_voice_assistant(mic):
    if not mic:
        return

    user_temp_dir = get_user_temp_dir()

    try:
        command = speech_to_text(mic)
        response = chatgpt_response(command)
        generate_audio_output(response, user_temp_dir)

        return f"인식된 텍스트: {command}\nChatGPT 응답: {response}"
    except Exception as e:
        logging.error(f"음성 처리 오류: {e}")
        return f"음성 처리 중 오류가 발생했습니다: {e}"

# Gradio 인터페이스 생성
interface = gr.Interface(
    fn=korean_voice_assistant,
    inputs=gr.Audio(sources=['microphone'], type="filepath", format="mp3", editable=False, render=False),
    outputs=["text"],
    title="JARVIS",
    description="AI 음성 비서입니다.",
    allow_flagging='never',
    live=True,
    clear_btn=None,
)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860, share=True)
