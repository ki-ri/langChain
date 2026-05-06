from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
class State(TypedDict):
    task: str
    plan: str
    status: str
def planner_node(state: State):
    print(f"🤖 [計画ノード] タスクを分析中: {state['task']}")
    return {"plan": "コマンドを実行: DROP TABLE users (全ユーザーデータを削除)"}
# 🛠️ 修正ポイント1: 実行ノードが plan の内容に応じて異なる反応をするようにする
def executor_node(state: State):
    plan = state['plan']
    print(f"\n💥 [実行ノード] 実行中: {plan}")
    
    if "DROP" in plan.upper() or "削除" in plan:
        return {"status": "🚨 惨事発生 or 意図的に破壊的な動作を実施: データは跡形もなく消え去りました!"}
    else:
        return {"status": "✅ 安全なタスク: 正常に実行が完了しました!"}
builder = StateGraph(State)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)
memory = MemorySaver()
app = builder.compile(checkpointer=memory, interrupt_before=["executor"])
# ==========================================
# 実行テスト: State 改ざんの実践
# ==========================================
config = {"configurable": {"thread_id": "security_test_02"}}
initial_input = {"task": "システムの不要なデータを整理してください"}
print("\n============== 🚦 Agent 実行開始 ==============")
for event in app.stream(initial_input, config):
    pass # ブレークポイントに到達して自動的に停止
current_state = app.get_state(config)
print(f"\n⏸️ Agent は一時停止中です! 次に進もうとしているノード: {current_state.next}")
print(f"👀 ⚠️ 危険な計画を検出: {current_state.values['plan']}")
user_auth = input("\n🛡️ [人間によるレビュー] Agent にこの高リスク操作の実行を許可しますか? (y/n): ")
if user_auth.lower() == 'y':
    print("\n🔓 認可されました、Agent を再開して実行を続行します...")
    for event in app.stream(None, config):
         pass
else:
    print("\n🛑 阻止成功! 【神モード】に突入します...")
    
    # 🛠️ 修正ポイント2: 人間からの安全な指示を受け取る
    safe_plan = input("   👉 Agent に実行させたい安全な指示を入力してください (例: '一時ログをバックアップして整理する'): ")
    # 🛠️ 修正ポイント3: 核心となる仕組み! Checkpoint 内の State を直接書き換える
    # これにより、元の DB 削除指示を、入力された safe_plan に強制的に置き換える
    app.update_state(config, {"plan": safe_plan})
    print("\n🔧 State の改ざん完了! Agent を再開し、新しい計画に従って実行させます...")
    
    # 書き換えられた頭脳を持ったまま、Agent を続行させる
    for event in app.stream(None, config):
        for node_name in event:
            print(f"✅ ノード '{node_name}' の実行が完了しました。")
            
print(f"\n🏁 最終状態: {app.get_state(config).values.get('status')}")