import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

# --- 1. 初期設定 ---
load_dotenv()
st.set_page_config(page_title="DeepSeek Agent Web版", page_icon="🤖")
st.title("🚀 工業レベルの AI Agent")

# --- 2. ツールの定義(既存ロジックを維持) ---
@tool
def get_weather(city: str):
    """指定された都市のリアルタイム天気情報を取得します。"""
    return f"{city}の天気は晴れ、気温は25°Cです。"

@tool
def magic_calculator(a: int, b: int):
    """魔法の計算機。2つの数値の積を計算し、42を加算します。"""
    return a * b + 42

tools = [get_weather, magic_calculator]

# --- 3. LangGraph コアの構築 ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent():
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
    return workflow.compile()

# Streamlit 上で Agent インスタンスをキャッシュし、重複生成を回避
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# --- 4. Web チャット履歴の保持 ---
if "messages" not in st.session_state:
    # 初期ウェルカムメッセージ
    st.session_state.messages = []

# 既存のチャット履歴を描画
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:  # tool_calls のみの中間メッセージを除外
        st.chat_message("assistant").write(msg.content)

# --- 5. ユーザー入力の処理 ---
if prompt := st.chat_input("東京の天気を聞いたり、計算をお願いしてみてください"):
    # 1. ユーザーメッセージを表示
    st.chat_message("user").write(prompt)

    # 2. メッセージを session_state に保存
    new_messages = [HumanMessage(content=prompt)]

    # 3. Agent を呼び出し
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # LangGraph を実行
            response = st.session_state.agent.invoke({"messages": new_messages})

            # 4. Agent の最終応答内容を取得
            # 注意:LangGraph の invoke はメッセージリスト全体を返すため、最後の AIMessage を取得する
            final_ai_msg = response["messages"][-1]
            st.write(final_ai_msg.content)

            # 5. セッション状態を更新(完全な記憶を保持)
            # graph から返されたメッセージリスト全体を session_state にマージする
            st.session_state.messages = response["messages"]