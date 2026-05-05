from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# ============================================================
# 1. 状態の拡張：summary フィールドを追加
#    メッセージリストに加えて、過去の対話の要約を保持するフィールドを持たせる
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

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
# 2. 対話ノード：現在のメッセージだけでなく、「日記（要約）」も参照する
# ============================================================
def chatbot_node(state: State):
    summary = state.get("summary", "")
    # 日記（要約）がある場合、システムプロンプトとしてモデルに渡す
    if summary:
        sys_msg = SystemMessage(content=f"これは以前の対話の要約です：{summary}")
        messages = [sys_msg] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)

    print(f"\n-----------> chatBot response: {response.content}")
    return {"messages": [response]}

# ============================================================
# 3. ルーティング関数（条件エッジ）：次にどこへ進むかを決定する
# ============================================================
def should_summarize(state: State):
    messages = state["messages"]
    # テスト環境：メッセージリストが4件を超えた場合（2ターン以上の対話）、要約ノードへ
    if len(messages) > 4:
        print("🚦 [ルーティング判定] メッセージが長すぎるため、バックグラウンドで要約を実行します → 'summarize_node'")
        return "summarize"
    print("🚦 [ルーティング判定] メッセージの長さは正常です。今回のターンを終了します → 'END'")
    return END

# ============================================================
# 4. 要約ノード：日記を書き、古いメッセージを削除する
# ============================================================
def summarize_node(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    # 日記を書くためのプロンプトを構築
    summary_prompt = (
        f"以下の新しい対話内容を簡潔な要約にまとめてください。\n"
        f"既に以前の要約がある場合は、新旧の要約を統合してください。\n\n"
        f"以前の要約: {summary}\n"
        f"新しい対話: {[m.content for m in messages]}"
    )
    # LLM に日記を書かせる
    response = llm.invoke(summary_prompt)

    # 【LangGraph のマジック】：RemoveMessage 命令を送信して、
    # 最後の2件（最新の質問と回答）以外の古いメッセージをすべて削除する
    # これにより、次のループ時には messages リストが再び短くなる！
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

    print(f"\n-----------> summarize_node response: {response.content}")
    print(f"\n-----------> summarize_node delete_messages: {delete_messages}")

    return {
        "summary": response.content,
        "messages": delete_messages
    }

# ============================================================
# 5. 複雑なグラフの組み立て
# ============================================================
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("summarize", summarize_node)

graph_builder.add_edge(START, "chatbot")

# 条件エッジの使用：chatbot の実行後、should_summarize 関数で次のステップを決定する
graph_builder.add_conditional_edges("chatbot", should_summarize)

# 要約完了後、今回のグラフ実行を終了する
graph_builder.add_edge("summarize", END)

# グラフ内で「メッセージ削除」操作を行うため、LangGraph はメモリチェックポイント（Checkpointer）の
# 有効化を要求する。ここでは最もシンプルなインメモリチェックポイント方式を使用する
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# ============================================================
# 6. 実行テスト
# ============================================================
print("\n============== 🧠 自動メモリクリーニング機能付き Agent が起動しました ==============")

# thread_id を設定して、この対話セッションのメモリを追跡する
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("\n👨‍💻 あなた: ")
    if user_input.lower() in ['q', 'quit']:
        break
    if not user_input.strip():
        continue

    # 実行順でメッセージを出力してフローを明確にする
    app.invoke({"messages": [("user", user_input)]}, config)
    # stream メソッドは各ノードの出力を表示するため、裏側のフローが明確にわかる
    # for event in app.stream({"messages": [("user", user_input)]}, config):
        # for node_name, node_state in event.items():
        #     if node_name == "chatbot":
        #         print(f"🤖 エージェント: {node_state['messages'][-1].content}")
        #     elif node_name == "summarize":
        #         print(f"\n📝 [バックグラウンド日記を更新しました]: {node_state['summary']}")