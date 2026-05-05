from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 定义状态
class State(TypedDict):
    task: str
    plan: str
    status: str

# 2. 计划节点（模拟大模型生成了危险指令）
def planner_node(state: State):
    print(f"🤖 [计划节点] 正在分析任务: {state['task']}")
    print("🤖 [计划节点] 思考完毕，生成执行计划...")
    return {"plan": "执行命令: DROP TABLE users (清空所有用户数据)"}

# 3. 执行节点（模拟真正去操作数据库）
def executor_node(state: State):
    print(f"\n💥 [执行节点] 正在执行: {state['plan']}")
    return {"status": "数据已灰飞烟灭！"}

# 4. 组装图
builder = StateGraph(State)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

# 5. 【核心魔法：设置断点】
memory = MemorySaver()
# interrupt_before=["executor"] 的意思是：在流向 executor 节点之前，无条件暂停！
app = builder.compile(checkpointer=memory, interrupt_before=["executor"])

# ==========================================
# 运行测试：见证“人在回路”
# ==========================================
config = {"configurable": {"thread_id": "security_test_01"}}
initial_input = {"task": "清理一下系统无用的数据"}

print("\n============== 🚦 Agent 开始运行 ==============")

# 第一阶段：Agent 开始运行，直到撞上断点
for event in app.stream(initial_input, config):
    # 我们只打印经过了哪些节点，不打印具体的细节以保持清爽
    for node_name in event:
        print(f"✅ 节点 '{node_name}' 执行完毕。")

# 第二阶段：Agent 已经被冻结，我们来查看它暂停时的状态
current_state = app.get_state(config)
print(f"\n⏸️ Agent 已暂停！它正准备进入下一个节点: {current_state.next}")
print(f"👀 ⚠️ 拦截到它的危险计划: {current_state.values['plan']}")

# 第三阶段：人工介入
user_auth = input("\n🛡️ [人工审核] 是否允许 Agent 执行此高危操作？(y/n): ")

if user_auth.lower() == 'y':
    print("\n🔓 授权通过，解冻 Agent，继续执行...")
    # 核心语法：传入 None 作为输入，告诉 LangGraph 顺着上次的断点继续往下跑
    for event in app.stream(None, config):
         for node_name in event:
            print(f"✅ 节点 '{node_name}' 执行完毕。")
    print(f"\n🏁 最终状态: {app.get_state(config).values.get('status')}")
else:
    print("\n🛑 授权拒绝！已成功阻止该危险操作。")
    # 在真实场景中，这里我们不会直接退出，而是会利用 update_state() 修改它的计划，让它重新思考。