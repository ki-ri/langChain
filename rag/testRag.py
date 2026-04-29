# ============================================================
# LangChain を使った RAG（検索拡張生成）の基本パイプライン
#
# 全体の流れ：
#   テキストファイル読み込み → チャンク分割 → ベクトル化 → ベクトルDB格納 → 類似度検索
#
# RAG とは：
#   LLM が回答する前に、関連するドキュメントを検索して
#   コンテキストとして渡すことで、正確な回答を生成する手法。
#   LLM の学習データに含まれない社内文書やリアルタイム情報にも対応できる。
# ============================================================

# --- インポート ---

# TextLoader: テキストファイルをLangChainのDocumentオブジェクトとして読み込むローダー
# Documentオブジェクトは page_content（本文）と metadata（メタ情報）を持つ
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter: 長い文書を小さなチャンクに分割するユーティリティ
# 段落 → 改行 → 文 → 文字 の順に再帰的に分割を試み、意味の切れ目を極力保つ
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# HuggingFaceEmbeddings: HuggingFace のモデルを使ってテキストをベクトル（数値配列）に変換する
# 意味的に近い文章は、ベクトル空間上でも近い位置にマッピングされる
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma: 軽量なベクトルデータベース。ベクトル化されたチャンクを格納し、
# クエリとの類似度検索（コサイン類似度など）を高速に実行できる
from langchain_community.vectorstores import Chroma

# 追加：コマンドライン引数を取得するための標準ライブラリ
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. テストデータの準備
#    同じディレクトリに test.txt がなければ、サンプルテキストを自動生成する
#    実運用では社内ドキュメント、FAQ、マニュアル等を読み込む想定
# ============================================================
file_path = "testRetrieval.txt"

if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            "LangChain は大規模言語モデル駆動のアプリケーション開発フレームワークです。\n"
            "アプリケーションにコンテキスト接続と推論能力を付与します。\n"
            "RAG はその中でも非常に重要な技術的方向性です..."
        )


# ============================================================
# 2. ドキュメントの読み込み（Load）
#    TextLoader はテキストファイルを丸ごと1つの Document として読み込む
#    docs はリストだが、1ファイル = 1要素なので docs[0] で本文全体にアクセスできる
# ============================================================
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

print(f"ファイルの読み込みに成功しました。現在は1つの文書全体で、長さは {len(docs[0].page_content)} 文字です")


# ============================================================
# 3. ドキュメントの分割（Split）
#
#    なぜ分割が必要か？
#    - LLM のコンテキストウィンドウには文字数制限がある
#    - 文書全体を渡すよりも、関連部分だけを渡した方が回答精度が上がる
#    - ベクトル検索は「小さな意味単位」で行う方がマッチング精度が高い
#
#    RecursiveCharacterTextSplitter のパラメータ：
#    - chunk_size: 各チャンクの最大文字数
#      （ここではデモ用に50と極小値。実運用では500〜1000が一般的）
#    - chunk_overlap: チャンク間の重複文字数
#      文の途中で切断されて文脈が失われるのを防ぐためのバッファ
#    - add_start_index: 各チャンクが元文書の何文字目から始まるかを
#      metadata に記録する（デバッグや出典表示に便利）
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    add_start_index=True
)

# split_documents() で実際に分割を実行
# 戻り値は Document オブジェクトのリスト（各チャンクが1つの Document）
splits = text_splitter.split_documents(docs)

print(f"\n文書が {len(splits)} 個のチャンク（Chunks）に分割されました：\n")
for i, chunk in enumerate(splits):
    print(f"--- チャンク {i+1} ---")
    print(chunk.page_content)
    # metadata には source（ファイルパス）と start_index（開始位置）が含まれる
    print(f"メタデータ (Metadata): {chunk.metadata}\n")


# ============================================================
# 4. エンベディングモデルの読み込み（Embedding Model）
#
#    エンベディングとは：
#    テキストを固定長の数値ベクトル（例：384次元の浮動小数点配列）に変換すること
#    「意味が近い文章 → ベクトル空間上で距離が近い」という性質を持つ
#
#    all-MiniLM-L6-v2:
#    - HuggingFace 提供の軽量な多言語対応モデル（約80MB）
#    - 出力次元: 384
#    - 初回実行時に自動ダウンロードされる
#    - 本番環境ではより高精度なモデル（例：multilingual-e5-large）を検討
# ============================================================
print("エンベディングモデルを読み込んでいます（初回実行時は自動ダウンロードされます。しばらくお待ちください）...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ============================================================
# 5. ベクトルDB への格納（Store）
#
#    from_documents() が内部で行う処理：
#      ① splits 内の各チャンク（Document）の page_content を取得
#      ② embeddings_model を使ってテキスト → ベクトルに変換
#      ③ ベクトルと元テキスト・メタデータをセットで Chroma DB に格納
#
#    Chroma はインメモリで動作するため、ここではプロセス終了時にデータが消える
#    永続化する場合は persist_directory パラメータを指定する
# ============================================================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings_model
)

print("✅ ドキュメントチャンクのベクトル化と Chroma データベースへの格納が完了しました！")


# .env ファイルから API キーと BASE URL を読み込む
load_dotenv()

# DeepSeek の LLM インスタンスを作成
# ChatOpenAI は OpenAI 互換 API であれば接続可能（DeepSeek もこの形式に対応）
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
)

# Chroma データベースを LangChain 標準の「リトリーバー」に変換
# リトリーバー = クエリを受け取り、関連ドキュメントを返すインターフェース
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# RAG 専用プロンプトの設計（モデルへの「オープンブック試験の問題用紙」）
# {context} に検索結果が、{question} にユーザーの質問が自動的に埋め込まれる
template = """あなたは知的アシスタントです。以下に提供された【コンテキスト】に厳密に基づいて、ユーザーの質問に回答してください。
コンテキストに関連情報が含まれていない場合は、「提供されたドキュメントに基づくと、わかりません」と正直に回答し、情報を捏造しないでください。

【コンテキスト】
{context}

【ユーザーの質問】
{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

# ユーティリティ関数：検索された複数のドキュメントチャンクのテキストを結合する
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL（LangChain Expression Language）で完全な RAG チェーンを構築
# パイプライン：
#   question → RunnablePassthrough() でそのまま渡す
#   context  → retriever で検索 → format_docs でテキストを結合
#   → prompt に埋め込み → LLM で回答生成 → StrOutputParser で文字列として出力
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# "このフレームワークは何に使えますか？"という質問を受け取った場合は下記のような prompt が生成される
# ============================================================
# あなたは知的アシスタントです。以下に提供された【コンテキスト】に厳密に基づいて...
#
# 【コンテキスト】
# LangChain は大規模言語モデル駆動のアプリケーション開発フレームワークです。
# アプリケーションにコンテキスト接続と推論能力を付与します。
#
# 【ユーザーの質問】
# このフレームワークは何に使えますか？
#
# 回答：
# ============================================================


# ============================================================
# 8. インタラクティブ対話システムの起動
#    無限ループでユーザーからの質問を受け付け、RAG チェーンで回答を生成する
#    'q' または 'quit' を入力すると終了
# ============================================================
print("\n============== インテリジェント RAG アシスタントが起動しました ==============")
while True:
    query = input("\n質問を入力してください（'q' または 'quit' で終了）: ")

    if query.lower() in ['q', 'quit']:
        print("システムを終了しました。")
        break
    if not query.strip():
        continue

    print("エージェントが資料を参照しながら考えています...")

    # rag_chain.invoke() を呼ぶだけで、検索 → プロンプト埋め込み → LLM 呼び出しが自動実行される
    response = rag_chain.invoke(query)

    print("\n--- エージェントの回答 ---")
    print(response)
