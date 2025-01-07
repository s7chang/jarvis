import os
import datetime
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import SerpAPIWrapper
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import playsound
import requests

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# STT (Speech-to-Text) 함수
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

# TTS (Text-to-Speech) 함수
def text_to_speech(output_text):
    tts = gTTS(text=output_text, lang="en")
    output_file = "output.mp3"
    tts.save(output_file)
    playsound.playsound(output_file)
    os.remove(output_file)

# 현재 시각 가져오기 함수
def get_current_time():
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%H:%M:%S')}."

# 현재 날씨 가져오기 함수
def get_weather(location="Seoul"):
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Weather API key is not set."
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temperature = data["main"]["temp"]
            weather = data["weather"][0]["description"]
            return f"The current weather in {location} is {weather} with a temperature of {temperature}°C."
        else:
            return f"Could not fetch weather information for {location}."
    except Exception as e:
        return f"An error occurred while fetching weather information: {e}"

# LangChain 구성: LLM, 메모리, 툴 추가
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key,
)
memory = ConversationBufferMemory()

# 웹 검색 툴 설정 (SerpAPI)
if not serpapi_api_key:
    print("SerpAPI key is not set. Web search functionality will be disabled.")
else:
    web_search_tool = SerpAPIWrapper(api_key=serpapi_api_key)

# 툴 정의
tools = [
    Tool(
        name="Current Time",
        func=lambda _: get_current_time(),
        description="Provides the current time."
    ),
    Tool(
        name="Weather",
        func=lambda location: get_weather(location),
        description="Provides the current weather for a given location. Use the location name as input."
    ),
]

# 웹 검색 툴 추가 (조건부)
if serpapi_api_key:
    tools.append(
        Tool(
            name="Web Search",
            func=web_search_tool.run,
            description="Searches the web for information on a given topic."
        )
    )

# 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Main 파이프라인
def main():
    while True:
        # 1. 음성 입력 -> 텍스트 변환 (STT)
        input_text = speech_to_text()
        if input_text is None:
            print("No valid input. Exiting.")
            break

        # 2. 에이전트 호출 (LangChain 툴 활용)
        try:
            response = agent.run(input_text)
            print(f"Agent response: {response}")
        except Exception as e:
            response = f"An error occurred while processing your request: {e}"
            print(response)

        # 3. 응답 -> 음성 출력 (TTS)
        text_to_speech(response)

if __name__ == "__main__":
    main()
