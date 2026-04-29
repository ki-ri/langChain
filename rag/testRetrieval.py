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

# コマンドライン引数から検索クエリを取得
# 使い方: python script.py "このフレームワークは何に使えますか？"
if len(sys.argv) < 2:
    print("使い方: python script.py \"検索したい質問文\"")
    sys.exit(1)

query = sys.argv[1]

print(f"\nユーザーの質問: {query}")

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


# ============================================================
# 6. 類似度検索の実行（Similarity Search）
#
#    処理の流れ：
#      ① ユーザーの質問文をエンベディングモデルでベクトル化
#      ② Chroma DB 内の全チャンクベクトルとの距離（コサイン類似度等）を計算
#      ③ 最も距離が近い（＝意味が最も似ている）チャンクを k 件返す
#
#    k=1 なので、最も関連性の高い1チャンクだけを取得
#    実際の RAG パイプラインでは、ここで取得したチャンクを
#    LLM のプロンプトに「参考情報」として埋め込み、回答を生成させる
# ============================================================
# query = "このフレームワークは何に使えますか？"


similar_docs = vectorstore.similarity_search(query, k=1)

print("\n--- 検索された最も関連性の高いドキュメントチャンク ---")
print(similar_docs[0].page_content)
print(f"ソースメタデータ: {similar_docs[0].metadata}")
