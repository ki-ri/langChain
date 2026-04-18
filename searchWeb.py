import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# 1. 初始化模型 (继续使用你的 DeepSeek 配置)
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
)

# 2. 初始化搜索工具 (设置只返回 2 条结果，节省 Token)
search_tool = TavilySearch(k=2)
tools = [search_tool]
model_with_tools = model.bind_tools(tools)

# 3. 执行逻辑
query = "2026年奥斯卡颁奖典已经结束，告诉我谁是 2026 年奥斯卡最佳男主角的得主？" # 这是一个 LLM 无法通过预训练知道的即时问题

# --- 第一轮：模型思考 ---
ai_msg = model_with_tools.invoke([HumanMessage(content=query)])

# --- 第二轮：执行搜索 ---
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # 运行搜索
        search_result = search_tool.invoke(tool_call["args"])
        print(f"搜索到的网页快照: {search_result}")

        # --- 第三轮：汇总回答 ---
        final_messages = [
            HumanMessage(content=query),
            ai_msg,
            ToolMessage(content=str(search_result), tool_call_id=tool_call["id"])
        ]
        final_answer = model_with_tools.invoke(final_messages)
        print(f"\n--- 最终答案 ---\n{final_answer.content}")
else:
    print(f"AI 觉得不需要搜索，直接回答了：{ai_msg.content}")
