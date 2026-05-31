import streamlit as st
import os
import sys
import asyncio
from dotenv import load_dotenv

# 导入核心库
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Annotated, TypedDict

# --- 1. 页面基本配置 ---
load_dotenv()
st.set_page_config(page_title="双轨 MCP 终极特工", page_icon="⚙️", layout="wide")
st.title("🎛️ 终极形态：多物理服务协同（Filesystem + SQLite）MCP Agent")

allowed_dir = os.path.abspath(".")

with st.sidebar:
    st.header("🌐 分布式服务集群状态")
    st.success("🟢 节点 1：Filesystem Server (Node.js 驱动)")
    st.success("🟢 节点 2：SQLite Database Server (Python FastMCP 驱动)")
    st.text(f"安全沙盒路径:\n{allowed_dir}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. 核心异步多服务驱动引擎 ---
async def run_cluster_agent(user_prompt: str):
    
    # 🌟 核心工程魔法：在这里同时配置两个风马牛不相及的服务
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
            "transport": "stdio",
        },
        "database": {
            # sys.executable 能够完美锁死当前 streamlit 正在运行的虚拟环境路径，避免多环境冲突报错
            "command": sys.executable, 
            "args": ["db_server.py"],
            "transport": "stdio",
        }
    }
    
    # 实例化聚合客户端
    client = MultiServerMCPClient(mcp_config)
    
    try:
        # 一键收割两个服务器上的所有原生工具！
        combined_tools = await client.get_tools()
        
        # 绑定给大模型
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0
        ).bind_tools(combined_tools)
        
        # 构建图
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            
        async def chatbot(state: State):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}
            
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot)
        workflow.add_node("tools", ToolNode(combined_tools, handle_tool_errors=True))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        agent_app = workflow.compile()
        
        st.session_state.messages.append(HumanMessage(content=user_prompt))
        
        message_placeholder = st.empty()
        full_response = ""
        
        events = agent_app.astream(
            {"messages": st.session_state.messages},
            stream_mode=["messages", "values"]
        )
        
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
            
    except Exception as e:
        st.error(f"🚨 集群控制台报错: {str(e)}")

# --- 3. 渲染聊天历史 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 4. 接收跨领域指令 ---
if prompt := st.chat_input("下达一个需要同时动用数据库和文件系统的终极任务？"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        asyncio.run(run_mcp_agent_macro := run_cluster_agent(prompt))