import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("API_KEY")

llm = ChatOpenAI(
	model="gpt-4o-mini",  # 사용할 OpenAI 모델
    temperature=0.7,  # 창의성 정도
    openai_api_key=openai_api_key,
)

response = llm.invoke("Hello, world!")
print(response)