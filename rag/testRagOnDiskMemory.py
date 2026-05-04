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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory


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
    chunk_size=500,
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

# ==========================================
# コアアップグレード領域：Agent に「海馬体（記憶機能）」を搭載する
#
# なぜ記憶が必要か？
#   前回のコードでは、毎回の質問が完全に独立していた。
#   例えば「LangChain って何？」と聞いた後に「それの主な機能は？」と聞くと、
#   「それ」が何を指すのか分からず、的外れな検索結果が返ってくる。
#
#   人間の会話では「それ」「あれ」「さっきの」といった指示語を自然に使うが、
#   LLM は各リクエストが独立しているため、会話の流れを自力では追えない。
#
# 解決策：2段階のパイプライン
#   ① 質問リライター：曖昧な質問を、会話履歴を踏まえて明確な質問に書き換える
#   ② Q&A チェーン：書き換えられた質問でドキュメント検索 → LLM が回答を生成
#
# ==========================================

# ----------------------------------------------------------
# ステップ1：「質問リライター」の作成（History-Aware Retriever）
#
# 目的：
#   ユーザーの曖昧な質問を、会話履歴を参照して自己完結した質問に変換する
#
# 具体例：
#   会話履歴: [ユーザー: "LangChainって何？", AI: "LLMアプリ開発フレームワークです"]
#   新しい質問: "それの主な機能は？"
#       ↓ リライト後
#   "LangChain の主な機能は何ですか？"
#
#   こうすることで、ベクトル検索が「LangChain の機能」に関する
#   正確なチャンクをヒットできるようになる。
#   「それ」のままだと、何を検索すべきか分からない。
# ----------------------------------------------------------

# システムプロンプト：LLM に「質問の書き換え役」としての役割を与える
# 重要なポイント：
#   - 「回答はしないでください」と明記 → LLM が勝手に回答し始めるのを防ぐ
#   - 「書き換えが不要な場合はそのまま返す」→ 既に明確な質問はそのまま通過させる
contextualize_q_system_prompt = """会話履歴と最新のユーザー質問が与えられます。\
最新の質問は会話履歴のコンテキストを参照している可能性があります。\
会話履歴がなくても理解できる、独立した完全な新しい質問に書き換えてください。\
質問の書き換えのみを行い、回答はしないでください。書き換えが不要な場合はそのまま返してください。"""

# ChatPromptTemplate.from_messages() でマルチターン対話用のプロンプトを構築
# 3つの要素で構成される：
#   ("system", ...)          → システムプロンプト（LLM の役割定義）
#   MessagesPlaceholder(...) → 会話履歴が動的に挿入される場所
#   ("human", "{input}")     → ユーザーの最新の質問
#
# Kotlin で例えると、以下のようなデータ構造のイメージ：
#   data class PromptInput(
#       val systemMessage: String,        // 固定のシステム指示
#       val chatHistory: List<Message>,   // 可変長の会話履歴
#       val userInput: String             // 今回の質問
#   )
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),  # 会話履歴を挿入するためのプレースホルダー
    ("human", "{input}"),
])

# create_history_aware_retriever() が行うこと：
#   ① chat_history が空の場合 → リライトをスキップし、そのまま retriever で検索
#   ② chat_history がある場合 → LLM で質問をリライト → リライト後の質問で検索
#
# 内部的には以下のパイプラインが構築される：
#   入力(質問 + 履歴) → [LLM で質問リライト] → [リライト後の質問で retriever 検索]
#                                                  → 関連ドキュメントを返す
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# ----------------------------------------------------------
# ステップ2：「最終Q&Aチェーン」の作成（QA Chain）
#
# 目的：
#   ステップ1で検索されたドキュメントと会話履歴を踏まえて、
#   LLM に最終的な回答を生成させる
#
# 以前の単純な RAG との違い：
#   以前 → context + question の2変数だけ
#   今回 → context + chat_history + input の3変数
#         会話履歴があることで、LLM は対話の流れを理解した上で回答できる
# ----------------------------------------------------------

# Q&A 用システムプロンプト
# {context} にはステップ1で検索されたドキュメントが自動的に埋め込まれる
qa_system_prompt = """あなたは知的アシスタントです。以下に提供された【コンテキスト】に厳密に基づいて、ユーザーの質問に回答してください。
コンテキストに関連情報が含まれていない場合は、「わかりません」と正直に回答し、情報を捏造しないでください。

【コンテキスト】
{context}"""

# Q&A 用プロンプトにも MessagesPlaceholder を含める
# これにより、LLM は以下の全情報を参照して回答を生成する：
#   - システム指示 + 検索結果（context）
#   - 過去の会話履歴（chat_history）
#   - ユーザーの最新の質問（input）
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),  # 最終回答時にも会話履歴を参照する
    ("human", "{input}"),
])

# create_stuff_documents_chain は LangChain が提供するユーティリティ関数
# 「stuff」= ドキュメントをプロンプトに「詰め込む」戦略
# 検索された全ドキュメントを結合して {context} に一括挿入する
# （ドキュメント量が多い場合は map_reduce や refine 等の別戦略もある）
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# ----------------------------------------------------------
# ステップ3：合体！記憶機能付き完全版 RAG チェーンの作成
#
# create_retrieval_chain() が2つのチェーンを直列に接続する：
#
#   ユーザー入力 + 会話履歴
#       │
#       ▼
#   ┌─────────────────────────────┐
#   │ history_aware_retriever     │  ← ステップ1
#   │ (質問リライト → 検索)        │
#   └─────────────┬───────────────┘
#                 │ 検索結果（ドキュメント）
#                 ▼
#   ┌─────────────────────────────┐
#   │ question_answer_chain       │  ← ステップ2
#   │ (検索結果 + 履歴 → 回答生成) │
#   └─────────────┬───────────────┘
#                 │
#                 ▼
#   最終回答（response["answer"]）
# ----------------------------------------------------------
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ==========================================
# 対話システムの起動と記憶管理
# ==========================================

print("\n============== 🧠 記憶機能付き RAG アシスタントが起動しました ==============")

# chat_history：Agent の「短期記憶メモ帳」
# HumanMessage と AIMessage のペアを時系列で蓄積していく
#
# データ構造のイメージ（3ターンの会話後）：
#   chat_history = [
#       HumanMessage("LangChain って何？"),              # 第1ターン：ユーザー
#       AIMessage("LLM アプリ開発フレームワークです"),     # 第1ターン：AI
#       HumanMessage("それの主な機能は？"),               # 第2ターン：ユーザー
#       AIMessage("コンテキスト接続と推論能力の付与です"), # 第2ターン：AI
#       HumanMessage("具体例を教えて"),                   # 第3ターン：ユーザー
#       AIMessage("RAG による検索拡張生成が代表例です"),   # 第3ターン：AI
#   ]
#
# 注意点：
#   この実装では会話が長くなるほどリストが無限に伸びていく。
#   本番環境では以下のような対策が必要：
#   - 直近 N ターンだけ保持する（スライディングウィンドウ）
#   - 古い履歴を要約して圧縮する
#   - トークン数の上限を設けて超過分を切り捨てる

# chat_history = []

# ----------------------------------------------------------
# 会話履歴の永続化：SQLite を使った長期記憶の実装
#
# 上記のインメモリ方式（chat_history = []）では、プロセスを終了すると
# 全ての会話履歴が失われてしまう。
#
# SQLChatMessageHistory を使うことで、会話履歴を SQLite データベースに
# 自動的に保存・復元できる。次回起動時にも過去の会話を覚えている。
#
# Kotlin/Java で例えると：
#   インメモリ方式 → ArrayList<Message>（プロセス終了で消える）
#   SQLite 方式   → Room / JPA でデータベースに永続化
# ----------------------------------------------------------

# 1. セッション履歴を取得する関数を定義
def get_session_history(session_id: str):
    # 実行ディレクトリに memory.db ファイルが自動作成される
    # session_id ごとに会話履歴が隔離されるため、
    # 複数ユーザーや複数の対話セッションを安全に管理できる
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///memory.db"
    )

# 2. セッション ID を設定（現在の対話セッションを識別するための ID）
# 実際のアプリケーションでは、ユーザー ID や対話ウィンドウの ID を使用する
session_id = "test_user_001"
chat_history_db = get_session_history(session_id)


while True:
    query = input("\n質問を入力してください（'q' で終了）: ")

    if query.lower() in ['q', 'quit']:
        print("システムを終了しました。")
        break
    if not query.strip():
        continue

    print("エージェントが資料を参照しながら考えています...")

    # 3. 运行对话时
    current_messages = chat_history_db.messages

    # 以前の invoke(query) と異なり、辞書形式で入力する
    # "input": 今回の質問
    # "chat_history": これまでの全会話履歴
    # → この2つが contextualize_q_prompt と qa_prompt の両方に渡される
    response = rag_chain.invoke({
        "input": query,
        "chat_history": current_messages
    })

    print("\n--- エージェントの回答 ---")
    # create_retrieval_chain の戻り値は辞書で、以下のキーを含む：
    #   "answer"  → LLM の最終回答テキスト
    #   "context" → 検索されたドキュメントのリスト（デバッグ用に確認可能）
    #   "input"   → 元の入力質問
    print(response["answer"])

    # 【極めて重要なステップ】：今回の質疑応答を記憶メモ帳に追加する
    #
    # なぜこれが重要か？
    #   この行がないと、次のターンで chat_history が空のままになり、
    #   質問リライターが「それ」「あれ」を解決できない。
    #   つまり、記憶機能が完全に無効になってしまう。
    #
    # add_user_message / add_ai_message は SQLChatMessageHistory のメソッドで、
    # HumanMessage / AIMessage として SQLite に自動保存される
    chat_history_db.add_user_message(query)
    chat_history_db.add_ai_message(response["answer"])
