import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering  import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader 

from pydantic import BaseModel, Field
from typing import List


# Define a estrutura de dados que queremos que o LLM retorne.
class CitedAnswer(BaseModel):
    """Uma resposta que cita suas fontes."""
    answer: str = Field(
        ...,
        description="A resposta à pergunta do usuário, baseada apenas nas fontes fornecidas."
    )
    source_indices: List[int] = Field(
        ...,
        description="A lista de índices (começando em 1) das fontes que foram usadas para formular a resposta."
    )


def configure_api():
    """Configura a API Key do Google de forma flexível."""
    # Prioridade 1: Tentar carregar dos segredos do Streamlit (para deploy no Streamlit Cloud)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
        return
    except (KeyError, FileNotFoundError):
        pass # Segredo não encontrado, vamos para a próxima tentativa

    # Prioridade 2: Tentar carregar de uma variável de ambiente (para Docker)
    try:
        api_key = os.environ["GOOGLE_API_KEY"]
        # Se a variável de ambiente já existe, não precisamos fazer nada.
        if not api_key: # Verifica se a variável não está vazia
             raise KeyError
    except KeyError:
        # Se não encontrou em nenhum lugar, exibe o erro e para.
        st.error("Chave de API do Google não encontrada. Configure st.secrets ou a variável de ambiente GOOGLE_API_KEY.")
        st.stop()
 
directory_path = 'data'
FAISS_INDEX_PATH = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
configure_api()

# --- Funções do RAG ---
def load_documents():
    """Carrega documentos de uma pasta usando TextLoader."""
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    return documents

def get_text_chunks(documents):
    """Divide os documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(_documents): # O underline evita que o Streamlit hasheie o objeto grande
    """Cria a vector store. Usamos cache para não recriar a cada interação."""
    text_chunks = get_text_chunks(_documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

 
def get_existing_vector_db():
    """Carrega o vector store já salvo em disco."""
    vector_db = FAISS.load_local(
    folder_path=FAISS_INDEX_PATH, 
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # Necessário para carregar os metadados dos documentos
    )

    return vector_db


# Função para criar o banco de dados vetorial
def create_vector_db_from_files():
    # Carrega os documentos
    documents = load_documents()
    # Cria o banco de dados vetorial com FAISS 
    vector_db = get_vector_store(documents)
    # Salva o banco de dados em disco
    vector_db.save_local(FAISS_INDEX_PATH)
    
    return vector_db

def get_response_from_conversational_chain(sources,_prompt):
    """Cria a cadeia de conversação com o LLM."""
    prompt_template = """
    Você é um assistente de IA avançado. Sua tarefa é responder à pergunta do usuário de forma precisa.
    Você receberá um contexto de documentos e uma pergunta. Siga esta regra de decisão:

    **REGRA DE DECISÃO:**
    1. Primeiro, avalie se o `Contexto de Documentos` fornecido contém informações relevantes para responder à `Pergunta`.
    2. **SE o contexto for relevante**, sua resposta DEVE ser baseada **exclusivamente** nas informações contidas nele. Cite as fontes que você usar.
    3. **SE o contexto NÃO for relevante** ou não for útil para responder à pergunta, você DEVE **ignorar completamente o contexto** e usar seu conhecimento geral para fornecer a melhor e mais completa resposta possível. Neste caso, não cite nenhuma fonte (a lista de fontes deve ser vazia).

    ---
    **Contexto de Documentos:**
    {context}
    ---
    **Pergunta:**
    {question}
    ---

    Lembre-se: priorize o contexto, mas não se limite a ele se não for relevante. Responda no formato solicitado.
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    structured_llm = model.with_structured_output(CitedAnswer)

    prompt_template  = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) 
    
    chain = prompt_template | structured_llm
    
    response_model = chain.invoke({
                        "context": context_with_sources,
                        "question": prompt
                    })
     
    return response_model

def format_docs_with_sources(docs: List[Document]) -> str:
    """Formata os documentos recuperados com um número de fonte."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Remove quebras de linha excessivas para um prompt mais limpo
        content_cleaned = " ".join(doc.page_content.split())
        formatted_docs.append(f"[Fonte {i+1} - {os.path.basename(doc.metadata['source'])}]:\n'{content_cleaned}'")
    return "\n\n".join(formatted_docs)

# --- Lógica da Aplicação Streamlit ---

st.set_page_config(page_title="Chat com Documentos", layout="wide")
st.header("📄 Chat com RAG, Gemini e Docker")

# Carrega e processa o documento de conhecimento (vector store)
if os.path.isdir(FAISS_INDEX_PATH):
    vector_store = get_existing_vector_db()
else:
    vector_store = create_vector_db_from_files()
    
  
# Inicializa o histórico do chat na sessão
if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Pergunte-me algo sobre os documentos."}]

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "cited_sources" in message and message["cited_sources"]:
            with st.expander("Fontes Utilizadas"):
                for source in message["cited_sources"]:
                    st.info(f"Fonte: {os.path.basename(source.metadata['source'])}\n\nTrecho: \"...{source.page_content}...\"")

# Captura a pergunta do usuário
if prompt := st.chat_input("Faça sua pergunta"):
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Gera e exibe a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Busca por documentos similares na vector store
            retrieved_docs = vector_store.similarity_search(prompt)
            
            # Formatar o contexto com as fontes numeradas
            context_with_sources = format_docs_with_sources(retrieved_docs)
            
            # Executa a cadeia de conversação
            response_model = get_response_from_conversational_chain(context_with_sources, prompt)
            
            # Quebra do retorno do modelo
            answer = response_model.answer
            cited_indices = response_model.source_indices
            
            # Filtra os documentos originais para obter apenas os citados
            cited_sources = [retrieved_docs[i-1] for i in cited_indices if 0 < i <= len(retrieved_docs)]

            # Exibe a resposta
            st.write(answer)
            
            # Exibe as fontes citadas, se houver
            if cited_sources:
                with st.expander("Fontes Utilizadas"):
                    for source in cited_sources:
                        st.info(f"Fonte: {os.path.basename(source.metadata['source'])}\n\nTrecho: \"...{source.page_content}...\"")
            
            # Adiciona ao histórico
            assistant_message = {
                "role": "assistant", 
                "content": answer, 
                "cited_sources": cited_sources
            }
            st.session_state.messages.append(assistant_message)
            