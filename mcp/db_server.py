# db_server.py
import sqlite3
import os
from mcp.server.fastmcp import FastMCP

# 1. 实例化 FastMCP，给它起个名字
mcp = FastMCP("sqlite-enterprise-server")

DB_PATH = "company_inventory.db"

# 2. 自动初始化一个本地 SQLite 数据库并塞入一些模拟数据
def init_mock_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 创建产品表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        price REAL,
        stock INTEGER
    )
    """)
    # 如果表是空的，塞入点科技公司的特产数据
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
        INSERT INTO products (name, price, stock) VALUES (?, ?, ?)
        """, [
            ("DeepSeek 算力扩容包 (100万Token)", 9.9, 999),
            ("量子纠缠防脱发洗发水", 199.0, 50),
            ("赛博朋克 2026 典藏版机甲手办", 1299.5, 5),
            ("高级 AI Agent 架构师毕业证书(镀金)", 0.1, 100)
        ])
        conn.commit()
    conn.close()

# 运行初始化
init_mock_database()

# 3. 通过 @mcp.tool() 装饰器，向外吐出标准工具
@mcp.tool()
def execute_database_query(sql_command: str) -> str:
    """当用户需要查询、增加、修改、删除公司产品库存数据库(company_inventory.db)中的内容时，调用此工具。
    支持标准的 SELECT, INSERT, UPDATE 等 SQL 语句。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_command)
        
        # 如果是查询语句，返回数据结构
        if sql_command.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            result = [dict(zip(columns, row)) for row in rows]
            conn.close()
            return f"📊 数据库执行成功，查询到以下结果：\n{str(result)}"
        else:
            # 如果是修改语句，提交事务
            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()
            return f"✅ 数据库修改成功，影响了 {affected_rows} 行数据。"
    except Exception as e:
        return f"❌ SQL 执行失败，报错原因: {str(e)}"

# 4. 启动 stdio 标准通信流
if __name__ == "__main__":
    mcp.run(transport="stdio")