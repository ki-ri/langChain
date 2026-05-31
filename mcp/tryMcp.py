import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

# 导入 LangChain, LangGraph 以及最核心的 MCP 适配器
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Annotated, TypedDict

# --- 1. 初始化页面配置 ---
load_dotenv()
st.set_page_config(page_title="MCP 文件特工", page_icon="🛠️", layout="wide")
st.title("🖲️ 2026 顶流标准：基于 MCP 协议的本地文件 Agent")

# --- 2. 确定 Agent 的安全活动目录 ---
# 我们把当前代码所在的绝对路径作为 Agent 的“沙盒限制区域”
allowed_dir = os.path.abspath(".")

with st.sidebar:
    st.header("⚙️ MCP 权限控制")
    st.info(f"📁 **允许操作的本地沙盒目录**:\n`{allowed_dir}`")
    st.caption("基于安全合规设计，Agent 无法越权读写该目录之外的任何系统文件。")

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 核心异步运行引擎 ---
async def run_mcp_agent(user_prompt: str):
    """全异步驱动的 MCP 智能体执行函数"""
    
    # 远程/本地连接配置：通过 stdio 管道动态拉起外部的 Node.js 官方文件系统 MCP 服务
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
            "transport": "stdio",
        }
    }
    
    # 创建 MCP 客户端
    client = MultiServerMCPClient(mcp_config)
    
    try:
        # 1. 核心魔术：一键拉取 MCP 服务器上所有的标准化工具（read_file, write_file, list_directory等）
        mcp_tools = await client.get_tools()
        
        # 2. 实例化大模型，并绑定这些外部标准工具
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0
        ).bind_tools(mcp_tools)
        
        # 3. 构建异步 LangGraph 工作流
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            
        async def chatbot(state: State):
            # 升级为异步 ainvoke 提升 I/O 性能
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}
            
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot)
        # 将 MCP 工具注入到工具节点中
        workflow.add_node("tools", ToolNode(mcp_tools, handle_tool_errors=True))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        agent_app = workflow.compile()
        
        # 4. 将用户新问题推入会话历史
        st.session_state.messages.append(HumanMessage(content=user_prompt))
        
        # 5. 准备流式渲染占位符
        message_placeholder = st.empty()
        full_response = ""
        
        # 开启图的异步流式输出 (astream)
        events = agent_app.astream(
            {"messages": st.session_state.messages},
            stream_mode=["messages", "values"]
        )
        
        final_state = None
        async for event_type, data in events:
            if event_type == "messages":
                chunk, metadata = data
                # 过滤并仅截获 chatbot 节点的文本输出
                if metadata.get("langgraph_node") == "chatbot" and chunk.content:
                    if isinstance(chunk.content, str):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
            elif event_type == "values":
                final_state = data
                
        # 移除光标，展现完美最终文本
        message_placeholder.markdown(full_response)
        
        # 6. 同步最终的图状态到 Streamlit
        if final_state is not None:
            st.session_state.messages = final_state["messages"]
            
    except Exception as e:
        st.error(f"🚨 MCP 运行时发生异常: {str(e)}")

# --- 4. 渲染网页前端历史聊天界面 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 5. 获取人类交互指令 ---
if prompt := st.chat_input("您可以命令我查看当前目录，或者创建、修改任意本地文件！"):
    # 在前端即时渲染用户输入
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        # 关键纽带：使用 asyncio.run 将 Streamlit 的同步主线程带入异步的 MCP 世界
        asyncio.run(run_mcp_agent(prompt))