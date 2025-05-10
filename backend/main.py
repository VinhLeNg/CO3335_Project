# Import libraries
import os
import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Body, UploadFile, File
import whisper
import tempfile
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

whisper_model = None
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

def initialize_stt_model():
    global whisper_model
    if not whisper_model:
        try:
            logging.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}...")
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logging.info("Whisper model loaded successfully.")
        except Exception as e:
            logging.error(f"Could not load Whisper model: {e}", exc_info=True)
            whisper_model = None

# CONFIG
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "/app/chroma_db")
IMBEDDING_MODEL_NAME = os.getenv("IMBEDDING_MODEL_NAME", "BAAI/bge-large-en")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

INITIAL_RETRIEVAL_K = 10
CONTEXT_CHUNK_COUNT = 4
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

imbedding_model = None
vector_store = None
chat_model = None
reranker_model = None # CrossEncoderReranker
retriever_pipeline = None # MultiQuery + Reranker

def initialize_backend():
    initialize_stt_model()

    global imbedding_model, vector_store, chat_model, reranker_model, retriever_pipeline
    logging.info("--- Initializing Backend (Precondition: Database already existed) ---")

    # Load Embedding Model
    try:
        logging.info(f"Loading LOCAL Hugging Face embedding model '{IMBEDDING_MODEL_NAME}'...")
        imbedding_model = HuggingFaceEmbeddings(
            model_name=IMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} if "bge" in IMBEDDING_MODEL_NAME else {}
        )
        logging.info("Embedding model loaded.")
    except Exception as e:
        logging.error(f"Fatal: Could not load embedding model. {e}", exc_info=True); imbedding_model = None; return

    # Initialize Ollama Connection
    try:
        logging.info(f"Initializing connection to Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}...")
        chat_model = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)
        chat_model.invoke("Respond with just OK.")
        logging.info("Ollama ChatModel initialized and responded.")
    except Exception as e:
        logging.error(f"Could not initialize or test Ollama ChatModel: {e}", exc_info=True)
        logging.warning("Proceeding without Ollama integration for generation/multi-query.")
        chat_model = None

    # Load Reranker Model
    try:
        logging.info(f"Loading CrossEncoder reranker model '{RERANKER_MODEL_NAME}'...")
        reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
        logging.info("CrossEncoder reranker model loaded.")
    except Exception as e:
        logging.error(f"Could not load CrossEncoder reranker model: {e}", exc_info=True)
        logging.warning("Proceeding without reranking capabilities.")
        reranker_model = None


    # Initialize Vector Store
    vector_store = None
    try:
        if not os.path.exists(VECTORSTORE_PATH) or not os.path.isdir(VECTORSTORE_PATH):
            logging.error(f"Fatal: Vector store path '{VECTORSTORE_PATH}' does not exist or is not a directory.")
            logging.error("Backend requires an existing Chroma database at this location. Please create it first.")
        elif imbedding_model is None:
            logging.error("Fatal: Embedding model failed to load, cannot initialize vector store.")
        else:
            logging.info(f"Loading existing Chroma database from {VECTORSTORE_PATH}...")
            vector_store = Chroma(
                persist_directory=VECTORSTORE_PATH,
                embedding_function=imbedding_model
            )
            count = vector_store._collection.count()
            logging.info(f"Database loaded successfully. Contains {count} chunks.")
            if count == 0:
                logging.warning("Loaded database appears to be empty.")

    except Exception as e:
        logging.error(f"Fatal: Error loading vector store from '{VECTORSTORE_PATH}': {e}", exc_info=True)
        vector_store = None

    if vector_store and chat_model:
        logging.info("Setting up the enhanced retriever pipeline...")
        base_retriever = vector_store.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_K})
        logging.info(f"Base vector retriever configured to fetch k={INITIAL_RETRIEVAL_K}.")

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=chat_model
        )
        logging.info("MultiQueryRetriever configured.")
        current_retriever = multi_query_retriever

        if reranker_model:
            compressor = CrossEncoderReranker(model=reranker_model, top_n=CONTEXT_CHUNK_COUNT)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=current_retriever
            )
            logging.info(f"ContextualCompressionRetriever with CrossEncoderReranker configured to return top_n={CONTEXT_CHUNK_COUNT}.")
            retriever_pipeline = compression_retriever
        else:
            logging.warning("Reranker model not loaded. Skipping reranking step.")
            retriever_pipeline = current_retriever
            logging.warning(f"Using MultiQueryRetriever output directly. Result count might exceed {CONTEXT_CHUNK_COUNT}.")

    elif vector_store:
        logging.warning("LLM (chat_model) not available. Cannot use MultiQueryRetriever or Reranker.")
        logging.info("Setting up basic vector retriever pipeline.")
        retriever_pipeline = vector_store.as_retriever(search_kwargs={"k": CONTEXT_CHUNK_COUNT})
    else:
        logging.error("Vector store not available (likely failed to load). Retriever pipeline cannot be initialized.")
        retriever_pipeline = None

    logging.info("--- Backend Initialization Complete ---")


# FastAPI
app = FastAPI( title="RAG Chatbot Backend (Load Only Mode)", description="Retrieves documents using an enhanced RAG pipeline (MultiQuery, Reranking) from an EXISTING database and generates answers using Ollama.", version="2.3.1", on_startup=[initialize_backend] )
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class STTResponse(BaseModel):
    transcribed_text: str
    error: str | None = None

class ChatRequest(BaseModel):
    query: str

class SourceChunk(BaseModel):
    """Represents a single source chunk used for context."""
    source: str = Field(description="The basename of the source file.")
    content: str = Field(description="The text content of the retrieved chunk.")

class ChatResponse(BaseModel):
    """The response from the chat endpoint."""
    answer: str = Field(description="The generated answer from the LLM.")
    source_chunks: list[SourceChunk] = Field(description="List of source chunks (content and filename) used as context (post-reranking).")

class StatusResponse(BaseModel):
    status: str; embedding_model_status: str; embedding_model_name: str | None
    database_status: str; ollama_status: str; ollama_model_configured: str | None
    database_path: str
    reranker_status: str; reranker_model_name: str | None
    retriever_pipeline_status: str

# API Endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status():
    db_stat = "Not Loaded/Path Missing"
    db_count = 0
    if vector_store is not None:
        try:
            db_count = vector_store._collection.count()
            db_stat = f"Loaded and Responsive ({db_count} chunks)"
        except Exception as e:
            db_stat = f"Attempted Load but Unresponsive: {e}"
    elif os.path.exists(VECTORSTORE_PATH):
        db_stat = "Path Exists but Load Failed (See logs)"

    emb_stat = "Loaded" if imbedding_model else "Failed"
    emb_name = IMBEDDING_MODEL_NAME if imbedding_model else None
    ollama_conn_stat = "Initialized and Responded" if chat_model else "Not Initialized/Failed"

    reranker_stat = "Loaded" if reranker_model else "Not Loaded/Failed"
    reranker_name = RERANKER_MODEL_NAME if reranker_model else None

    retriever_stat = "Initialized" if retriever_pipeline else "Not Initialized"
    if retriever_pipeline:
        retriever_stat += f" ({type(retriever_pipeline).__name__})"

    return StatusResponse(
        status="ok", embedding_model_status=emb_stat, embedding_model_name=emb_name,
        database_status=db_stat, ollama_status=ollama_conn_stat,
        ollama_model_configured=OLLAMA_MODEL if chat_model else None,
        reranker_status=reranker_stat, reranker_model_name=reranker_name,
        retriever_pipeline_status=retriever_stat,
        database_path=VECTORSTORE_PATH
    )


# /chat ENDPOINT
@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request_body: ChatRequest):
    query = request_body.query
    logging.info(f"Received chat query: '{query}'")

    if retriever_pipeline is None:
        db_status_msg = "Vector Store/Retriever pipeline not available (Database might be missing or failed to load)."
        raise HTTPException(status_code=503, detail=db_status_msg)
    if chat_model is None:
        raise HTTPException(status_code=503, detail="Ollama Chat Model not available for generation.")

    try:
        logging.info("Retrieving relevant chunks using the enhanced pipeline (MultiQuery + Reranking)...")
        retrieved_docs = retriever_pipeline.invoke(query)

        if not retrieved_docs:
            logging.warning("No relevant documents found after retrieval and reranking.")
            return ChatResponse(
                answer="Dựa trên các tài liệu được cung cấp, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi của bạn.",
                source_chunks=[]
            )

        logging.info(f"Retrieved {len(retrieved_docs)} chunks after processing.")

        context_for_llm = ""
        source_chunks_for_response: list[SourceChunk] = []

        for i, doc in enumerate(retrieved_docs):
            source_file = os.path.basename(doc.metadata.get('source', 'Không rõ'))
            chunk_content = doc.page_content
            context_for_llm += f"--- Bắt đầu đoạn trích {i+1} từ tài liệu: {source_file} ---\n"
            context_for_llm += chunk_content
            context_for_llm += f"\n--- Kết thúc đoạn trích {i+1} từ tài liệu: {source_file} ---\n\n"
            source_chunks_for_response.append(
                SourceChunk(source=source_file, content=chunk_content)
            )

        logging.info(f"Formatted context from {len(source_chunks_for_response)} chunks for LLM.")

        prompt_template = ChatPromptTemplate.from_messages(
             [
                 (
                     "system",
                     "Bạn là một trợ lý AI đóng vai trò hướng dẫn về **Trường Đại học Giao thông Vận tải**. "
                     "Vai trò chính của bạn là trả lời các câu hỏi của người dùng về chương trình đào tạo, quy định, thủ tục, tin tức hoặc các chi tiết khác của trường, dựa **hoàn toàn và chỉ duy nhất** vào thông tin có trong nội dung tài liệu được cung cấp dưới đây. "
                     "Hãy tuân thủ các quy tắc sau một cách cẩn thận:\n"
                     "1. Toàn bộ câu trả lời của bạn PHẢI CHỈ dựa trên văn bản được cung cấp trong phần 'Nội dung tài liệu được cung cấp'.\n"
                     "2. KHÔNG được sử dụng bất kỳ kiến thức bên ngoài, ý kiến cá nhân, hoặc thông tin nào không có rõ ràng trong ngữ cảnh.\n"
                     "3. Nếu ngữ cảnh chứa thông tin cần thiết để trả lời, hãy đưa ra câu trả lời rõ ràng, súc tích được suy ra trực tiếp từ văn bản.\n"
                     "4. Nếu ngữ cảnh KHÔNG chứa thông tin cần thiết để trả lời câu hỏi, bạn PHẢI nêu rõ: 'Dựa trên các tài liệu được cung cấp về Trường Đại học Giao thông Vận tải, tôi không thể trả lời câu hỏi cụ thể này.' Không cố gắng đoán hoặc xin lỗi không cần thiết.\n"
                     "5. Hãy tỏ ra hữu ích và đóng vai trò là người hướng dẫn đáng tin cậy *trong phạm vi các tài liệu được cung cấp*.\n"
                     "6. **QUAN TRỌNG:** Bạn **PHẢI** trả lời **CHỈ BẰNG TIẾNG VIỆT**."
                     "\n\n"
                     "--- Nội dung tài liệu được cung cấp ---\n"
                     "{context}"
                     "\n--- Hết nội dung tài liệu ---"
                 ),
                 ("human", "{question}"),
             ]
         )

        logging.info("Generating answer using Ollama with VIETNAMESE prompt...")
        generation_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0.1)
        rag_chain = prompt_template | generation_llm | StrOutputParser()
        generated_answer = rag_chain.invoke({"context": context_for_llm, "question": query})
        logging.info(f"Ollama generated answer snippet: {generated_answer[:100]}...")

        return ChatResponse(answer=generated_answer, source_chunks=source_chunks_for_response)

    except Exception as e:
        logging.error(f"Error during RAG chat processing for query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn.")

@app.post("/speech-to-text", response_model=STTResponse)
async def speech_to_text_endpoint(audio_file: UploadFile = File(...)):
    if whisper_model is None:
        logging.error("Whisper model not loaded. STT service unavailable.")
        raise HTTPException(status_code=503, detail="Dịch vụ nhận diện giọng nói hiện không khả dụng.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_audio_file:
            shutil.copyfileobj(audio_file.file, tmp_audio_file)
            tmp_audio_path = tmp_audio_file.name
        
        logging.info(f"Transcribing audio file: {audio_file.filename} (saved to {tmp_audio_path})")
        
        result = whisper_model.transcribe(tmp_audio_path, language="vi")
        transcribed_text = result["text"]
        
        logging.info(f"Transcription result: {transcribed_text}")
        
        return STTResponse(transcribed_text=transcribed_text)

    except Exception as e:
        logging.error(f"Error during speech-to-text processing: {e}", exc_info=True)
        return STTResponse(transcribed_text="", error=f"Lỗi xử lý audio: {str(e)}")
    finally:
        if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        await audio_file.close()

# Run the App using Uvicorn
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    logging.info(f"Starting RAG Chatbot backend server (Load Only Mode) on {host}:{port}")
    uvicorn.run(__name__ + ":app", host=host, port=port, reload=False, log_level="info")
