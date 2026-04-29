from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 准备一个测试用的本地文件 (假设你在同级目录下建了一个 test.txt)
# 你可以在里面随便贴几段新闻或者维基百科的内容
file_path = "test.txt"

if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("LangChain 是一个用于开发由大语言模型驱动的应用程序的框架。\n它使得应用程序能够连接上下文并进行推理。\nRAG 是其中非常重要的一个技术方向...")

# 2. 加载文档 (Load)
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()
print(f"成功加载文件，目前这是一个整体文档，长度为：{len(docs[0].page_content)} 个字符")

# 3. 切分文档 (Split)
# RecursiveCharacterTextSplitter 是最推荐的文本切分器，它会尽量保证段落和句子的完整性
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,       # 每个知识块最大 50 个字符 (为了演示效果设得很小，实际通常设为 500-1000)
    chunk_overlap=10,    # 块与块之间重叠 10 个字符，防止一句话被从中间硬生生切断，丢失上下文
    add_start_index=True # 记录每个小块在原文档中的起始位置
)

# 执行切分
splits = text_splitter.split_documents(docs)

print(f"\n文档被切分成了 {len(splits)} 个小块 (Chunks)：\n")
for i, chunk in enumerate(splits):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print(f"元数据 (Metadata): {chunk.metadata}\n")


# 4. 加载嵌入模型 (Embedding Model)
# 这里我们使用 HuggingFace 提供的一个轻量级开源中文/多语言友好模型
print("正在加载 Embedding 模型（首次运行会自动下载，请稍候）...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. 将文档块转化为向量，并存入 Chroma 向量数据库
# from_documents 会自动遍历 splits 里的每个小块，调用模型转成向量，然后存起来
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings_model
)
print("✅ 成功将文档块转化为向量，并存入 Chroma 数据库！")

# 6. 见证奇迹的时刻：执行相似度检索 (Similarity Search)
query = "这个框架能用来干什么？"
print(f"\n用户提问: {query}")

# k=1 表示寻找 1 个与问题在数学空间上距离最近的文档块
similar_docs = vectorstore.similarity_search(query, k=1)

print("\n--- 检索到的最相关文档块 ---")
print(similar_docs[0].page_content)
print(f"来源元数据: {similar_docs[0].metadata}")
