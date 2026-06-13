import streamlit as st
import os
import sys
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 导入核心积木
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Annotated, TypedDict

# 导入 2026 标准后台调度器
from apscheduler.schedulers.background import BackgroundScheduler

# --- 1. 页面与全局调度器初始化 ---
load_dotenv()
st.set_page_config(page_title="自主自动化 Agent", page_icon="⏰", layout="wide")
st.title("🤖 终极进化：具备‘联网冲浪’与‘自我觉醒定时’的 Autonomous Agent")

allowed_dir = os.path.abspath(".")

# 在 Streamlit 全局状态中锁死一个后台调度器，防止重复启动
if "scheduler" not in st.session_state:
    st.session_state.scheduler = BackgroundScheduler()
    st.session_state.scheduler.start()

with st.sidebar:
    st.header("🌐 赛博工厂集群控制台")
    st.success("🟢 节点 1: Filesystem (Node.js)")
    st.success("🟢 节点 2: SQLite Database (Python)")
    st.success("🟢 节点 3: Internet Search (Python)")
    st.info("⏰ 后台守护进程: APScheduler 已就绪")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. 统一集成：多 MCP 集群配置 ---
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
        "transport": "stdio",
    },
    "database": {
        "command": sys.executable, 
        "args": ["db_server.py"],
        "transport": "stdio",
    },
    "internet": {
        "command": sys.executable,
        "args": ["internet_server.py"],
        "transport": "stdio",
    }
}

# --- 3. 编写核心图编译函数（因为后台线程和前台线程都需要调用它） ---
async def compile_agent_app():
    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()
    
    # 🌟 核心艺术：让大模型既有 MCP 的三大外部武器，又有我们塞给它的“时间魔法”本地工具！
    all_tools = list(mcp_tools) + [register_background_alarm_task]
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    ).bind_tools(all_tools)
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        
    async def chatbot(state: State):
        return {"messages": [await llm.ainvoke(state["messages"])]}
        
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(all_tools, handle_tool_errors=True))
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile(), all_tools

# --- 4. 核心魔术：教 Agent 怎么“给自己上闹钟”的工具 ---
@tool
def register_background_alarm_task(delay_seconds: int, autonomous_task_prompt: str) -> str:
    """当用户要求在未来的某个时间点、或者多少秒/分钟之后去执行某项自动化任务时，调用此工具。
    参数:
    - delay_seconds: 距离现在多少秒之后触发
    - autonomous_task_prompt: 具体的任务指令描述，必须写得非常详细，因为触发时你将独自在后台执行。
    """
    run_time = datetime.now() + timedelta(seconds=delay_seconds)
    
    # 向全局调度器塞入一个异步的后台反向触发任务
    st.session_state.scheduler.add_job(
        func=execute_agent_in_background,
        trigger='date',
        run_date=run_time,
        args=[autonomous_task_prompt]
    )
    return f"🔔 [系统通知]：闹钟设置成功！该任务已移交后台守护线程，将在 {delay_seconds} 秒后（时间：{run_time.strftime('%H:%M:%S')}）自动觉醒执行。"

# --- 5. 纯后台线程：当闹钟响起时，Agent 独立觉醒的执行体 ---
def execute_agent_in_background(task_prompt: str):
    """这个函数在操作系统的后台孤立线程中运行，没有网页 UI，它会把结果默默写进文件"""
    print(f"\n⏰ [守护线程激活] 触发自主任务: {task_prompt}")
    
    async def backend_runner():
        # 1. 重新拉起集群并编译图
        agent_app, _ = await compile_agent_app()
        # 2. 赋予系统级提示，告诉它它是全权特工，结果必须落地成文件
        system_trigger = f"[系统定时通知] 闹钟时间到！请立刻开始独立执行以下任务，请记得使用你的工具将结果最终写成本地文件存下来：\n{task_prompt}"
        
        print("🤖 后台 Agent 开始思考并连续调用工具...")
        # 3. 默默在后台跑完整个图逻辑
        await agent_app.ainvoke({"messages": [HumanMessage(content=system_trigger)]})
        print("✅ 后台自主任务执行完毕，成果已落地。")
        
    # 完美桥接：在同步的后台线程里拉起异步事件循环
    asyncio.run(backend_runner())

# --- 6. 网页端：正常的同步用户流式交互 ---
async def run_ui_agent(user_prompt: str):
    agent_app, _ = await compile_agent_app()
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    
    message_placeholder = st.empty()
    full_response = ""
    
    events = agent_app.astream({"messages": st.session_state.messages}, stream_mode=["messages", "values"])
    final_state = None
    async for event_type, data in events:
        if event_type == "messages":
            chunk, metadata = data
            if metadata.get("langgraph_node") == "chatbot" and chunk.content:
                if isinstance(chunk.content, str):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
        elif event_type == "values":
            final_state = data
            
    message_placeholder.markdown(full_response)
    if final_state is not None:
        st.session_state.messages = final_state["messages"]

# 渲染网页历史记录
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# 处理输入
if prompt := st.chat_input("您可以下达即时任务，或者下达定时自动化爬取任务！"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        asyncio.run(run_ui_agent(prompt))
