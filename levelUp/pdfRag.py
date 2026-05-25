import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# 导入 LangChain & LangGraph 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

# --- 1. 初始化页面配置 ---
load_dotenv()
st.set_page_config(page_title="RAG Agent 稳定版", page_icon="📚", layout="wide")
st.title("🧙‍♂️ 拥有私有知识库的超级 Agent (架构优化版)")

# --- 2. 初始化会话状态 (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- 3. 侧边栏：处理 PDF 上传与解析 ---
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传你的 PDF 说明书/文档", type=["pdf"])
    
    if uploaded_file is not None:
        # 如果是新上传的文件，进行解析
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("正在挑灯夜读，拼命解析 PDF 中..."):
                # 1. 写入临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # 2. 解析与切分
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    splits = text_splitter.split_documents(docs)
                    
                    # 3. 构建内存检索器 (BM25 算法，无需 Embedding 密钥，纯内存运行)
                    retriever = BM25Retriever.from_documents(splits)
                    retriever.k = 3  # 每次检索最相关的 3 个片段
                    
                    # 4. 写入状态
                    st.session_state.retriever = retriever
                    st.session_state.current_file = uploaded_file.name
                    st.success(f"📚 《{uploaded_file.name}》解析成功！Agent 已熟读此书。")
                except Exception as e:
                    st.error(f"解析文件失败: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path) # 清理临时文件
    else:
        # 用户清空了上传的文件
        st.session_state.retriever = None
        st.session_state.current_file = None
        st.info("💡 请先上传一份 PDF，让 Agent 获得专属领域知识。")

# --- 4. 核心魔术：从主线程提取 Retriever 变量 ---
# 这一步非常关键！current_retriever 是一个标准 Python 变量
current_retriever = st.session_state.retriever

# --- 5. 动态定义工具箱 (利用闭包安全访问知识库) ---

@tool
def get_weather(city: str):
    """查询指定城市的实时天气情况。"""
    return f"{city}天气晴朗，气温 25°C，非常舒适。"

@tool
def query_uploaded_pdf(query: str) -> str:
    """当用户询问关于上传的 PDF 文档、报告、内部资料或私有知识库中的具体内容时，必须调用此工具。"""
    # 避开 st.session_state，直接访问主线程捕获的 current_retriever
    if current_retriever is None:
        return "错误：用户当前还没有上传任何 PDF 文档。请礼貌地提醒用户先在左侧边栏上传文档，否则你无法回答此问题。"
    
    try:
        print(f"\n query contents: {query}")
        # 执行检索
        docs = current_retriever.invoke(query)
        if not docs:
            return "在文档中没有找到与该问题直接相关的核心片段。"
        
        # 拼接知识片段
        context = "\n---\n".join([d.page_content for d in docs])
        return f"从用户上传的 PDF 文档中找到了以下相关核心参考片段：\n{context}"
    except Exception as e:
        return f"检索文档时发生错误: {str(e)}"

# 将工具打包
tools = [get_weather, query_uploaded_pdf]

# --- 6. 动态构建 LangGraph 智能体 ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 每次页面渲染时动态编译图，确保工具箱里的闭包始终绑定最新的 current_retriever
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
).bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")
agent_app = workflow.compile()

# --- 7. 渲染网页聊天历史 ---
# 过滤掉中间的工具调用消息，只给人类看文本对话
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 8. 处理用户输入与 Agent 运行 ---
if prompt := st.chat_input("您可以问我天气，或者问关于已上传 PDF 的任何问题！"):
    # 1. 在 UI 上展现用户输入
    st.chat_message("user").write(prompt)
    
    # 2. 先把当前用户的提问追加到记忆库中
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # 3. 呼叫 Agent 思考
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 将包含新问题的完整历史发送给 LangGraph
            response = agent_app.invoke({"messages": st.session_state.messages})
            
            # 4. 更新全局记忆（LangGraph 会把新产生的 AiMessage 和 ToolMessage 追加进去）
            st.session_state.messages = response["messages"]
            
            # 5. 取出最后一条消息（即 Agent 总结出的最终拟人回答）渲染到网页上
            final_ai_msg = response["messages"][-1]
            st.write(final_ai_msg.content)