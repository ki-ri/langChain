# internet_server.py
import sys
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

# 初始化联网 MCP 服务器
mcp = FastMCP("internet-search-server")

@mcp.tool()
def search_the_live_web(query: str) -> str:
    """当用户询问当前最新的实时新闻、近期热点、瞬息万变的市场股票、
    或者大模型本身不知道的 2026 年最新互联网知识时，必须调用此工具。
    """
    try:
        # 使用 DuckDuckGo 进行实时联网检索
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return "网络搜索完成，但未找到相关的公开网页结果。"
        
        # 格式化输出给大模型看
        formatted_results = []
        for r in results:
            formatted_results.append(f"📰 标题: {r['title']}\n🔗 链接: {r['href']}\n📝 摘要: {r['body']}\n---")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"❌ 联网搜索时发生网络抖动或错误: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")

