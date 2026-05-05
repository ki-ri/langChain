from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# 1. 状態を定義
class State(TypedDict):
    task: str
    plan: str
    status: str
# 2. 計画ノード（大規模モデルが危険な命令を生成したと想定）
def planner_node(state: State):
    print(f"🤖 [計画ノード] タスクを分析中: {state['task']}")
    print("🤖 [計画ノード] 思考完了、実行計画を生成...")
    return {"plan": "コマンドを実行: DROP TABLE users (全ユーザーデータを削除)"}
# 3. 実行ノード（実際にデータベースを操作することを想定）
def executor_node(state: State):
    print(f"\n💥 [実行ノード] 実行中: {state['plan']}")
    return {"status": "データは跡形もなく消え去りました！"}
# 4. グラフを組み立てる
builder = StateGraph(State)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)
# 5. 【核心となる仕組み:ブレークポイントを設定】
memory = MemorySaver()
# interrupt_before=["executor"] の意味は: executor ノードへ流れる前に無条件で一時停止する!
app = builder.compile(checkpointer=memory, interrupt_before=["executor"])
# ==========================================
# 実行テスト:「Human-in-the-Loop」を体感する
# ==========================================
config = {"configurable": {"thread_id": "security_test_01"}}
initial_input = {"task": "システムの不要なデータを整理してください"}
print("\n============== 🚦 Agent 実行開始 ==============")
# 第1段階: Agent が実行を開始し、ブレークポイントに到達するまで進む
for event in app.stream(initial_input, config):
    # どのノードを通過したかだけを出力し、詳細は省略してすっきり見せる
    for node_name in event:
        print(f"✅ ノード '{node_name}' の実行が完了しました。")
# 第2段階: Agent はすでに停止しているので、停止時の状態を確認する
current_state = app.get_state(config)
print(f"\n⏸️ Agent は一時停止中です! 次に進もうとしているノード: {current_state.next}")
print(f"👀 ⚠️ 危険な計画を検出: {current_state.values['plan']}")
# 第3段階: 人間による介入
user_auth = input("\n🛡️ [人間によるレビュー] Agent にこの高リスク操作の実行を許可しますか? (y/n): ")
if user_auth.lower() == 'y':
    print("\n🔓 認可されました、Agent を再開して実行を続行します...")
    # 核となる文法: None を入力として渡すことで、LangGraph に前回のブレークポイントから続行するよう伝える
    for event in app.stream(None, config):
         for node_name in event:
            print(f"✅ ノード '{node_name}' の実行が完了しました。")
    print(f"\n🏁 最終状態: {app.get_state(config).values.get('status')}")
else:
    print("\n🛑 認可が拒否されました! 危険な操作の阻止に成功しました。")
    # 実際の場面では、ここで直接終了するのではなく、update_state() を使って計画を修正し、再度思考させる。