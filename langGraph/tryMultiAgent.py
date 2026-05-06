from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 1. State の定義: 今回の状態は2つの Agent が共同編集する「共有ドキュメント」
class State(TypedDict):
    topic: str             # ユーザーから与えられたトピック
    draft: str             # 作業者が書いた下書き
    critique: str          # レビュアーからの修正意見
    revision_count: int    # 修正回数を記録(無限ループによる Token 枯渇を防ぐため)

# 2. 大規模モデルの準備
# .env ファイルから API キーと BASE URL を読み込む
load_dotenv()

# DeepSeek の LLM インスタンスを作成
# ChatOpenAI は OpenAI 互換 API であれば接続可能（DeepSeek もこの形式に対応）
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
)

# 3. ノード A: 作業者 Agent
def writer_node(state: State):
    current_count = state.get('revision_count', 0)
    print(f"\n✍️  [作業者 Agent] 鋭意執筆中... ({current_count + 1} 回目の執筆)")
    
    # 修正意見の有無に応じて、Prompt を動的に調整する
    if state.get('critique'):
        prompt = f"【{state['topic']}】に関する短文(約100字)を書き直してください。\n上司からの修正意見: {state['critique']}\n上司の要求を必ず満たしてください!"
    else:
        prompt = f"【{state['topic']}】に関する魅力的な短文(約100字)を書いてください。"

    response = llm.invoke(prompt)
    
    return {
        "draft": response.content,
        "revision_count": current_count + 1
    }

# 4. ノード B: レビュアー/上司 Agent
def reviewer_node(state: State):
    print("🧐 [レビュアー Agent] 老眼鏡をかけて下書きを審査中...")
    
    prompt = f"""あなたは非常に厳格な記事レビュアーです。以下の【{state['topic']}】に関する下書きを審査してください:
    
    【下書きの内容】
    {state['draft']}

    【レビュー回数】
    {state['revision_count']}
    
    【審査基準】
    1. 非常に生き生きとして面白い必要がある。
    2. レビュー回数が１回目の場合は指摘してください
    3. 下書きに比喩表現が含まれていない場合は、差し戻すこと。
    
    指摘がある場合は完璧ではないです。
    完璧だと思う場合は、"ACCEPT"(全て大文字)とだけ返答してください。
    何か不適切な点があると思う場合は、修正が必要な箇所を直接指摘してください(ACCEPT とは返答しないこと)。"""

    response = llm.invoke(prompt)
    critique = response.content



    if "ACCEPT" in critique.upper() and len(critique) < 15:
        print("✅ [レビュアー Agent] 合格としよう、文章は完璧だ!")
        return {"critique": "ACCEPT"}
    else:
        print(f"❌ [レビュアー Agent] 差し戻して書き直し! 意見: {critique}")
        return {"critique": critique}

# 5. ルーティング関数(条件付きエッジ): 書き直しを続けるか、終業して終わるかを決定する
def route_to_next(state: State):
    # 上司が満足したら、グラフの実行を終了する
    if state.get("critique") == "ACCEPT":
        return END
    # 貧乏人保護機構: 最大3回まで、これ以上は上司が破産する
    if state.get("revision_count", 0) >= 3:
        print("⚠️ [システム警告] 最大修正回数(3回)に到達、作業者の苦しみを強制終了します。")
        return END
    
    # それ以外は、作業者に差し戻して書き直させる
    return "writer"

# 6. 複雑なループグラフを組み立てる
builder = StateGraph(State)
builder.add_node("writer", writer_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "writer")      # 開始 -> 作業者
builder.add_edge("writer", "reviewer") # 作業者が書き終えたら -> レビュアーに渡す
builder.add_conditional_edges("reviewer", route_to_next) # レビュアーが見終えたら -> ルーティング判定

app = builder.compile()

# ==========================================
# 実行テスト: バーチャル会社を起動
# ==========================================
print("\n============== 🏢 Multi-Agent バーチャル会社、開業しました ==============")
initial_input = {"topic": "なぜプログラマーはコーヒーを好むのか?"}

# 最終成果物の表示
final_state = app.invoke(initial_input)
print(final_state.get('draft'))