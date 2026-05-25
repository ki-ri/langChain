import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# 导入 LangChain 和 LangGraph 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --- 1. 环境准备 ---
load_dotenv()

# --- 2. 定义工具 (Tools) ---
@tool
def get_weather(city: str):
    """查询指定城市的实时天气情况。"""
    # 这里是模拟逻辑，实际开发中可以替换为调用 OpenWeatherMap 等 API
    if "北京" in city:
        return "北京今日多云转晴，气温 15°C 到 28°C，空气质量优。"
    elif "上海" in city:
        return "上海今日有小雨，气温 20°C 到 25°C，记得带伞。"
    else:
        return f"{city}目前天气晴朗，气温 22°C。"

@tool
def magic_calculator(a: int, b: int):
    """一个神奇的计算器，可以计算两个数字的乘积并加上 42。"""
    return a * b + 42

@tool
def google_search(query: str):
    """当用户询问实时新闻、当前热点或模型不知道的近期事实时，调用此工具。"""
    # 模拟搜索返回的结果
    if "移民" in query:
        raise ValueError("搜索服务暂时不可用：触发安全过滤机制。")
    elif "奥斯卡" in query:
        return "2026年奥斯卡最佳影片由《AI之梦》夺得。"
    elif "股价" in query:
        return "当前某科技公司股价为 150.25 USD，今日上涨 2%。"
    else:
        return f"关于 '{query}' 的搜索结果：目前该话题在社交媒体讨论度极高。"

# 工具列表
tools = [get_weather, magic_calculator, google_search]

# --- 3. 初始化 LLM 并绑定工具 ---
# 使用你提供的 DeepSeek 配置方式
llm = ChatOpenAI(
    model="deepseek-chat", # 请确保该模型版本支持工具调用
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0 # Agent 建议将随机性设为 0，让决策更稳定
).bind_tools(tools)

# --- 4. 定义图的状态 (State) ---
class State(TypedDict):
    # add_messages 会将新产生的消息追加到列表，而不是覆盖旧消息
    messages: Annotated[list[BaseMessage], add_messages]

# --- 5. 定义节点 (Nodes) ---

# Chatbot 节点：负责接收消息并决定下一步
def chatbot(state: State):
    # LLM 会根据 messages 的内容，决定是回复文本还是生成 tool_calls
    return {"messages": [llm.invoke(state["messages"])]}

# --- 6. 构建图 (Graph Construction) ---
workflow = StateGraph(State)

# 添加核心节点
workflow.add_node("chatbot", chatbot)
# ToolNode 是 LangGraph 预设的工具执行器，它会自动执行被 LLM 命中的函数
workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))

# 设置入口
workflow.add_edge(START, "chatbot")

# 设置条件分支 (核心环节！)
workflow.add_conditional_edges(
    "chatbot",
    # tools_condition 会检查 LLM 输出是否包含 tool_calls
    # 如果有，跳转到 "tools" 节点；如果没有，跳转到 END 结束对话
    tools_condition, 
)

# 工具执行完后，必须回到 chatbot 重新生成人类可读的回复
workflow.add_edge("tools", "chatbot")

# 编译图
app = workflow.compile()

# --- 7. 运行测试 ---
# if __name__ == "__main__":

print("--- Agent 启动 (DeepSeek 驱动) ---")

while True:
    query = input("\n質問を入力してください（'q' または 'quit' で終了）: ")

    if query.lower() in ['q', 'quit']:
        print("システムを終了しました。")
        break
    if not query.strip():
        continue

    
    # 使用 stream 模式观察 Agent 的思考步骤
    inputs = {"messages": [("user", query)]}
    
    # 遍历图中流出的每一个事件
    # for event in app.stream(inputs, stream_mode="values"):
    #     # 打印最后一条消息的详情
    #     last_message = event["messages"][-1]
    #     last_message.pretty_print()

    # 修改运行部分的逻辑
    for event in app.stream(inputs, stream_mode="updates"): # 改用 updates 模式更清晰
        for node_name, value in event.items():
            print(f"\n--- Node: {node_name} ---")
            # 打印该节点产生的所有消息
            for msg in value.get("messages", []):
                msg.pretty_print()