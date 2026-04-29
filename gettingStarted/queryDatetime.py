import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from datetime import datetime

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

# 4. 使用tool
@tool
def get_current_time():
    """获取当前系统的时间。当用户询问‘现在几点’或相关时间问题时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 把工具放入列表
tools = [get_current_time]

# 绑定工具
model_with_tools = model.bind_tools(tools)

# 1. 用户提问
query = "帮我查一下现在的时间，并告诉我明天是几号？"

# 2. 模型思考：它会发现自己需要调用 get_current_time
ai_msg = model_with_tools.invoke(query)

# 打印一下 AI 的返回，你会发现它没有直接回答，而是产生了一个 tool_calls
print(f"AI 的意图: {ai_msg.tool_calls}")

# 3. 执行工具（模拟 Agent 的行为）
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # 寻找对应的工具并运行
        selected_tool = {"get_current_time": get_current_time}[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        
        print(f"工具执行结果: {tool_output}")
        
        # 4. 把工具结果喂回给 AI，让它给出最终答案
        # 构造发给 AI 的完整对话历史
        messages = [
            HumanMessage(content=query), # 用户的原问题
            ai_msg,                      # AI 刚才要调用工具的请求
            ToolMessage(                 # 工具执行的结果
                content=str(tool_output), 
                tool_call_id=tool_call["id"]
            )
        ]
        
        # 再次调用模型
        final_answer = model_with_tools.invoke(messages)
        print(f"最终答案: {final_answer.content}")
