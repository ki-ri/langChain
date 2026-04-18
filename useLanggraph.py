# ============================================================
# 基于 LangGraph 构建一个带搜索能力的 AI Agent
# 整体架构：用户提问 → LLM 判断是否需要搜索 → 搜索 → LLM 总结回答
# ============================================================

import os
from dotenv import load_dotenv

# --- 导入：类型标注工具 ---
# Annotated: 允许给类型附加额外元信息（这里用来绑定 reducer 函数）
# TypedDict: 类似 Kotlin 的 data class，定义一个有固定字段的字典类型
from typing import Annotated, TypedDict

# --- 导入：LangGraph 图构建核心组件 ---
# StateGraph: 状态图的构造器，用来定义节点和边
# START / END: 图的入口和出口的特殊标记节点
from langgraph.graph import StateGraph, START, END

# add_messages: 一个 reducer 函数，作用是"追加"而非"覆盖"消息列表
# 类似 Kotlin 中 list + newList 的效果，保证对话历史不断累积
from langgraph.graph.message import add_messages

# ToolNode: 预置的工具执行节点，接收 LLM 输出的工具调用请求并实际执行
# tools_condition: 预置的条件判断函数，检查 LLM 的回复中是否包含工具调用
#   - 如果有 → 走向 tools 节点
#   - 如果没有 → 走向 END 节点，直接输出结果
from langgraph.prebuilt import ToolNode, tools_condition

# --- 导入：LLM 和搜索工具 ---
# ChatOpenAI: LangChain 对 OpenAI 兼容 API 的封装（这里实际连接 DeepSeek）
from langchain_openai import ChatOpenAI

# TavilySearchResults: 基于 Tavily API 的网络搜索工具
# k=2 表示每次搜索返回 2 条结果
from langchain_tavily import TavilySearch


# ============================================================
# 1. 定义状态（State）
#    整个图在运行过程中共享的数据结构，所有节点都读写同一个 State
#    目前只有一个字段 messages，用来存储完整的对话历史
#    Annotated[list, add_messages] 的含义：
#      类型是 list，但每次更新时使用 add_messages 策略（追加而非覆盖）
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. 初始化模型和工具
#    ChatOpenAI(model="deepseek-chat") → 创建一个 DeepSeek 的 LLM 实例
#    .bind_tools([...]) → 告诉模型"你可以使用这些工具"
#    这样模型在回答时，如果发现需要查资料，会生成工具调用请求
#    而不是直接编造答案
# ============================================================
load_dotenv()
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
)

search_tool = TavilySearch(k=2)
tools = [search_tool]
model_with_tools = model.bind_tools(tools)


# ============================================================
# 3. 定义节点（Node）
#    节点是图中的处理单元，接收当前 State，返回需要更新的字段
#    chatbot 节点：将当前对话历史发给 LLM，获取回复
#    返回 {"messages": [...]} 会被 add_messages reducer 追加到现有列表中
# ============================================================
def chatbot(state: State):
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# ============================================================
# 4. 构建图（Graph）
#    定义节点之间的连接关系，形成完整的工作流
#
#    流程示意：
#
#    START → chatbot ──(需要工具)──→ tools ──→ chatbot（循环）
#                │
#                └──(不需要工具)──→ END
#
#    这个循环使得 Agent 可以多轮调用工具后再给出最终回答
# ============================================================
workflow = StateGraph(State)

# 注册节点：给每个处理函数起一个名字
workflow.add_node("chatbot", chatbot)                        # LLM 对话节点
workflow.add_node("tools", ToolNode([TavilySearch(k=2)]))  # 工具执行节点

# 定义边（连接关系）
workflow.add_edge(START, "chatbot")          # 入口：用户消息先进入 chatbot

# 条件边（核心逻辑）：
# tools_condition 会检查 chatbot 的输出中是否包含 tool_calls
#   - 有 tool_calls → 路由到 "tools" 节点去执行搜索
#   - 没有 tool_calls → 路由到 END，结束流程
workflow.add_conditional_edges("chatbot", tools_condition)

workflow.add_edge("tools", "chatbot")        # 工具执行完毕后，回到 chatbot
                                              # 让 LLM 根据搜索结果生成最终回答


# ============================================================
# 5. 编译并运行
#    compile() 将图定义转换为可执行的应用
#    invoke() 传入初始状态，触发整个流程
# ============================================================
app = workflow.compile()

# 测试：这个问题需要搜索才能回答（LLM 训练数据可能没有 2026 年的信息）
# 执行流程：
#   1. chatbot 收到问题 → 判断需要搜索 → 生成 tool_call
#   2. tools 节点执行搜索 → 返回搜索结果
#   3. chatbot 收到搜索结果 → 整理并生成最终回答
#   （如果信息不够，可能会再次搜索，形成多轮循环）
final_state = app.invoke({
    "messages": [("user", "2026 年奥斯卡最佳男主角已经确定，告诉我是谁，并简述他的成名作")]
})

# 取最后一条消息（即 LLM 的最终回答）并打印
print(final_state["messages"][-1].content)