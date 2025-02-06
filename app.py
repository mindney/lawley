import logging
import os
import datetime
import pickle
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_pdf(pdf_path):
    """Carga un archivo PDF y devuelve su contenido."""
    if not os.path.exists(pdf_path):
        logging.error(f"Archivo no encontrado: {pdf_path}")
        raise FileNotFoundError(f"El archivo '{pdf_path}' no existe.")
    
    logging.info(f"Cargando PDF: {pdf_path}")
    try:
        loader = PDFPlumberLoader(pdf_path)
        return loader.load()
    except Exception as e:
        logging.error(f"Error cargando PDF: {e}")
        return []

def process_document(docs, embedder, cache_name):
    """Procesa documentos y guarda los chunks en caché."""
    cache_path = f"{cache_name}.pkl"
    if os.path.exists(cache_path):
        logging.info(f"Cargando caché de {cache_name}...")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    
    logging.info(f"Dividiendo {cache_name} en chunks semánticos...")
    text_splitter = SemanticChunker(embedder)
    chunks = text_splitter.split_documents(docs)

    with open(cache_path, 'wb') as cache_file:
        pickle.dump(chunks, cache_file)
    
    return chunks

def create_vector_store(documents, embedder, index_name, num_documents=10):
    """Crea un índice FAISS para recuperación de contexto."""
    logging.info(f"Creando índice FAISS para {index_name}...")
    index_path = f"faiss_{index_name}"
    
    if os.path.exists(index_path):
        try:
            vector = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
            logging.info(f"Índice FAISS cargado: {index_name}")
        except Exception as e:
            logging.warning(f"Error cargando FAISS {index_name}: {e}. Reconstruyendo...")
            vector = FAISS.from_documents(documents, embedder)
            vector.save_local(index_path)
    else:
        vector = FAISS.from_documents(documents, embedder)
        vector.save_local(index_path)

    return vector.as_retriever(search_type="similarity", search_kwargs={"k": num_documents})

def configure_llm():
    """Configura el modelo de IA para análisis legal."""
    logging.info("Cargando modelo LLM...")
    return ChatOpenAI(
        model="o1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.05,
        max_tokens=100000,
        verbose=True
    )


def analyze_sentence(sentence_retriever, codes_retriever, llm):
    """Realiza el análisis legal con contexto persistente."""
    prompt_template = PromptTemplate(
        template="""
        **Objective:**
        Identify any errors, inconsistencies, contradictions, or any elements that could help in appealing the sentence in the most solid and effective manner. Use the included legal context for comparisons to enhance the ability to find and be more effective.
        Cite specific phrases from the sentence or the legal context to support and justify any analysis found.
        The response should be in Spanish

        **Legal Context:**
        {context_codes}
        
        **Analyzed Sentence:**
        {context_sentence}
        """,
        input_variables=["context_codes", "context_sentence"]
    )

    logging.info("Recuperando contexto de la sentencia...")
    sentence_context = sentence_retriever.invoke("Analiza esta sentencia")
    codes_context = codes_retriever.invoke("Proporciona fundamentos legales")
    
    response = llm.invoke(
        prompt_template.format(
            context_codes=codes_context,
            context_sentence=sentence_context
        )
    )

    generate_markdown_report(response.content)

def generate_markdown_report(content):
    """Genera un informe en Markdown."""
    os.makedirs("./reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = f"./reports/result_{timestamp}.md"
    
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(content)
    
    logging.info(f"El informe ha sido guardado en '{md_path}'")

def main():
    try:
        sentence_docs = load_pdf("./documents/sentencia.pdf")
        legal_docs = load_pdf("./documents/codigo-civil.pdf") + load_pdf("./documents/codigo-tributario.pdf")
        
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        sentence_chunks = process_document(sentence_docs, embedder, "sentencia")
        legal_chunks = process_document(legal_docs, embedder, "codigos_legales")
        
        sentence_retriever = create_vector_store(sentence_chunks, embedder, "sentencia")
        codes_retriever = create_vector_store(legal_chunks, embedder, "codigos_legales")
        
        llm = configure_llm()

        analyze_sentence(sentence_retriever, codes_retriever, llm)
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
