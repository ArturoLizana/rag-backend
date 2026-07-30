import os
from typing import Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

FAISS_INDEX_PATH = "faiss_index"

class RAGService:
    def __init__(self):
        self._embeddings = None
        self.vector_store: FAISS = None

    @property
    def embeddings(self):
        """Chargement du modèle uniquement quand une requête le demande."""
        if self._embeddings is None:
            print("⏳ Chargement du modèle d'embeddings HuggingFace...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Modèle d'embeddings prêt !")
        return self._embeddings

    def load_existing_index(self):
        """Recharge l'index FAISS sauvegardé sur le disque si présent."""
        if self.vector_store is None and os.path.exists(FAISS_INDEX_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Nécessaire pour FAISS local
                )
                print(" Index FAISS existant rechargé avec succès !")
            except Exception as e:
                print(f" Impossible de charger l'index existant : {e}")

    def _get_llm(self):
        api_key = settings.GROQ_API_KEY
        if not api_key or api_key == "gsk_ta_cle_groq_ici":
            raise ValueError("Veuillez configurer GROQ_API_KEY dans le fichier .env !")
        return ChatGroq(
            groq_api_key=api_key,
            model_name=settings.LLM_MODEL,
            temperature=0.2
        )

    def process_pdf(self, file_path: str) -> int:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Génération et sauvegarde locale de FAISS
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(FAISS_INDEX_PATH)
        print(f"💾 Index FAISS sauvegardé dans {FAISS_INDEX_PATH}")
        
        return len(chunks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        # Charger l'index existant si pas encore fait en mémoire
        if self.vector_store is None:
            self.load_existing_index()

        if not self.vector_store:
            return {
                "answer": "Veuillez d'abord télécharger un document PDF.",
                "sources": []
            }

        llm = self._get_llm()
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])

        # Extraction des sources (page + extrait)
        sources = []
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1
            sources.append({
                "page": page_num,
                "snippet": doc.page_content[:180] + "..."
            })
        
        prompt_template = ChatPromptTemplate.from_template("""
Tu es un assistant IA professionnel. Réponds à la question uniquement en t'appuyant sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis poliment que tu ne sais pas d'après le document.

Contexte :
{context}

Question : {question}

Réponse :
""")
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": sources
        }

rag_service = RAGService()