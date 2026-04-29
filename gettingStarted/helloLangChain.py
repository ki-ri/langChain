import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 加载 .env 文件中的环境变量
load_dotenv() 

# 2. 从环境变量中读取（这样代码里就不会出现明文 Key）
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

# 3. 初始化 DeepSeek 模型
model = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_key=api_key, 
    openai_api_base=base_url
)

# 4. 测试运行
prompt = ChatPromptTemplate.from_template("请给我讲一个关于 {topic} 的冷笑话")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

try:
    print(chain.invoke({"topic": "程序员"}))
except Exception as e:
    print(f"调用失败，请检查 Key 是否正确。错误信息：{e}")
