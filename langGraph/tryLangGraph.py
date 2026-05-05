from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# ============================================================
# 1. 状態（State）の定義
#    状態はグラフ全体を流れる「血液」のようなもの。
#    ここでは最も基本的な状態を定義する：対話メッセージのみを含む
# ============================================================
class State(TypedDict):
    # Annotated と add_messages の組み合わせが非常に重要：
    # ノードが新しいメッセージを返したとき、古いメッセージを「上書き」するのではなく、
    # リストに「追加」するようグラフに指示する。
    messages: Annotated[list, add_messages]

# ============================================================
# 2. LLM の初期化
# ============================================================
# .env ファイルから API キーと BASE URL を読み込む
load_dotenv()

# DeepSeek の LLM インスタンスを作成
# ChatOpenAI は OpenAI 互換 API であれば接続可能（DeepSeek もこの形式に対応）
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
)

# ============================================================
# 3. ノード（Node）の定義
#    ノードとは、State を受け取り、状態の更新内容を返す Python 関数のこと
# ============================================================
def chatbot_node(state: State):
    print("🤖 [内部ログ] chatbot_node に入りました。LLM を呼び出しています...")
    response = llm.invoke(state["messages"])
    # 返された内容は state["messages"] に追加される
    return {"messages": [response]}

# ============================================================
# 4. グラフ（Graph）の組み立て
# ============================================================
graph_builder = StateGraph(State)

# 作成した関数をノードとしてグラフに追加する
graph_builder.add_node("chatbot", chatbot_node)

# エッジ（Edges）の定義：データの流れる方向を規定する
graph_builder.add_edge(START, "chatbot")  # 開始点から chatbot ノードに直接流れる
graph_builder.add_edge("chatbot", END)    # chatbot ノードの実行後、終了点に到達して終了

# 実行可能な Agent アプリケーションにコンパイルする
app = graph_builder.compile()

# ============================================================
# 5. 実行テスト
# ============================================================
print("\n============== ⚡ LangGraph 基本スケルトンが起動しました ==============")

# 1回の単一ターン対話をシミュレーションする
user_input = "こんにちは、一言で褒めてください。"
print(f"👨‍💻 ユーザー: {user_input}\n")

# app.stream はグラフの実行をトリガーし、
# 各ノードを通過した後の状態変化をリアルタイムで返す
for event in app.stream({"messages": [("user", user_input)]}):
    for node_name, node_state in event.items():
        print(f"✨ [ノード {node_name} 実行完了] エージェントの返答: {node_state['messages'][-1].content}")