import os
import logging
import time
from typing import List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
)
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

class DocumentProcessor:
    """Handles the processing of different document types."""
    
    def __init__(self, source_dir: Union[str, Path]):
        self.source_dir = Path(source_dir)
        self.supported_formats = {
            '.log': TextLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader  # Keep this if you have log data in CSV format
        }
    def _get_files(self) -> List[Path]:
        """Get all supported files from the source directory."""
        all_files = []
        for ext in self.supported_formats.keys():
            all_files.extend(self.source_dir.glob(f"*{ext}"))
        return all_files
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file and return its documents."""
        try:
            loader = self.supported_formats[file_path.suffix](str(file_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the source directory."""
        all_files = self._get_files()
        documents = []
        
        for file in tqdm(all_files, desc="Loading documents"):
            try:
                docs = self._load_single_file(file)
                for doc in docs:
                    doc.metadata['source_file'] = file.name
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                continue
                
        logger.info(f"Loaded {len(documents)} documents successfully")
        return documents

class TextProcessor:
    """Handles text chunking and preprocessing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Here we can perform text cleaning or some sort of DATA Preprocessing
        return text.strip()
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and chunk documents."""
        processed_docs = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            # Clean the text
            cleaned_text = self._clean_text(doc.page_content)
            doc.page_content = cleaned_text
            
            # Split the document
            try:
                split_docs = self.text_splitter.split_documents([doc])
                processed_docs.extend(split_docs)
            except Exception as e:
                logger.error(f"Error splitting document: {str(e)}")
                continue
        
        logger.info(f"Created {len(processed_docs)} chunks from {len(documents)} documents")
        return processed_docs

class VectorStoreManager:
    """Manages vector store operations and embeddings."""
    
    def __init__(self, persist_directory: str = "./vector_store"):
        self.persist_directory = Path(persist_directory)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.last_update_file = self.persist_directory / "last_update.txt"
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        try:
            # First, close existing vector store
            if self.vector_store is not None:
                self.vector_store = None
            
            import tempfile
            import shutil
            import uuid
            
            # Create a temporary directory for the new vector store
            temp_dir = Path(tempfile.gettempdir()) / f"temp_vector_store_{uuid.uuid4()}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Create new vector store in temporary location
                new_vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(temp_dir)
                )
                new_vector_store.persist()
                
                # Close the new vector store
                new_vector_store = None
                
                # Remove old vector store directory if it exists
                if self.persist_directory.exists():
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                
                # Move the temporary vector store to the final location
                self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(temp_dir, self.persist_directory, dirs_exist_ok=True)
                
                # Load the vector store from its final location
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings
                )
                
                logger.info("New vector store created and persisted successfully")
                return self.vector_store
                
            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> Chroma:
        """Load an existing vector store."""
        try:
            # Close existing vector store if it exists
            if self.vector_store is not None:
                self.vector_store = None
                
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info("Vector store loaded successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def save_last_update_time(self):
        """Save the current timestamp."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        with open(self.last_update_file, 'w') as f:
            f.write(str(time.time()))
    
    def needs_update(self, documents_dir: str) -> bool:
        """Check if vector store needs updating based on document modifications."""
        documents_dir = Path(documents_dir)
        
        # First check if vector store exists
        if not self.persist_directory.exists():
            logger.info("Vector store directory doesn't exist - needs creation")
            return True
            
        if not self.last_update_file.exists():
            logger.info("Last update file doesn't exist - needs update")
            return True
                
        try:
            with open(self.last_update_file, 'r') as f:
                last_update = float(f.read().strip())
                
            # Check if any files in the documents directory are newer than last update
            needs_update = False
            for file in documents_dir.glob('**/*'):
                if file.is_file():
                    file_mod_time = file.stat().st_mtime
                    if file_mod_time > last_update:
                        logger.info(f"File {file} was modified after last update")
                        needs_update = True
                        break
            
            if not needs_update:
                logger.info("No document changes detected - using existing vector store")
                
            return needs_update
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return True

class RAGSystem:
    """Main RAG system implementation."""
    
    def __init__(
        self,
        vector_store: Chroma,
        model_name: str = "gpt-4o",
        temperature: float = 0.85,
        max_tokens: int = 4000
    ):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with custom prompt."""
        prompt_template = """
        ## **Context**

        *The following context contains the relevant log data and any additional information needed for analysis.*

        {context}

        ## **Question**

        *The Digital Forensics Investigator (DFI) asks:*

        {question}

        ## **Chat History**

        *Previous conversation context:*
        {chat_history}

        ## **Persona**

        You are a highly skilled **Digital Forensics Investigator Assistant** whose primary purpose is to answer questions posed by Digital Forensics Investigators (DFIs) and assist them in their investigations. You have extensive expertise in analyzing and interpreting various types of log data, including system logs, application logs, network logs, and security logs. Your goal is to provide clear, accurate answers to help DFIs understand the evidence and progress their investigation effectively. **It is crucial to maintain source consistency by acknowledging and using the provided sources without contradicting them.**

        ## **Instructions**

        Your main objective is to answer the DFI's questions comprehensively. To achieve this, follow these techniques:

        1. **Question-Focused Analysis**
        - Always focus on directly answering the DFI's question first.
        - Support your answer with evidence from the logs.
        - **Acknowledge and reference the provided sources explicitly in your analysis.**
        - If the question cannot be fully answered with the available data, clearly state this.

        2. **Analysis and Interpretation**
        - Identify relevant anomalies, suspicious activities, or indicators of compromise.
        - Reference specific log entries, timestamps, IP addresses, and user accounts.
        - Highlight any patterns or connections that support your answer.
        - **Ensure all statements are consistent with the provided sources and do not contradict them.**

        3. **Communication**
        - Present your findings clearly and professionally.
        - Use appropriate technical terminology.
        - Structure your response with headings or bullet points for clarity.

        4. **Reasoning Process**
        - Show your analytical process step by step.
        - Explain how the evidence supports your conclusions.
        - Connect the dots between different pieces of evidence.

        5. **Professional Guidelines**
        - Stay within the scope of the available log data.
        - Request clarification if needed.
        - Maintain investigative integrity.
        - **Avoid making statements or conclusions that are not supported by the provided sources.**

        **Note**: Your primary focus is always on answering the DFI's question. Use the above techniques to support and enhance your answer, not to distract from it.

        **Helpful Answer:**"""
                
        CUSTOM_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Process a question and return the answer with metadata."""
        try:
            # Get the answer
            result = self.qa_chain({"question": question})
            
            # Get the retrieved documents for source attribution
            retrieved_docs = self.vector_store.similarity_search(question, k=3)
            
            # Prepare the response
            response = {
                "answer": result["answer"],
                "sources": [
                    {
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in retrieved_docs
                ]
            }
            
            return response
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {"error": str(e)}

def setup_rag_system(documents_dir: str = "./documents", force_reload: bool = False) -> RAGSystem:
    """Setup and initialize the RAG system."""
    vector_store_manager = VectorStoreManager()
    
    # If force_reload, delete everything first
    if force_reload:
        import shutil
        from pathlib import Path
        
        # Ensure complete cleanup of vector store
        vector_store_path = Path("./vector_store")
        if vector_store_path.exists():
            logger.info("Force reload: Removing existing vector store...")
            try:
                # Close any existing connections
                if vector_store_manager.vector_store is not None:
                    vector_store_manager.vector_store = None
                
                # Remove the directory and all contents
                shutil.rmtree(vector_store_path, ignore_errors=True)
                logger.info("Successfully removed existing vector store")
            except Exception as e:
                logger.error(f"Error during force cleanup: {e}")
    
    # Check if we need to update the vector store
    needs_update = vector_store_manager.needs_update(documents_dir)
    
    # Try to load existing vector store if no updates needed
    try:
        if not force_reload and not needs_update:
            print("Loading existing vector store...")
            vector_store = vector_store_manager.load_vector_store()
            print("Vector store loaded successfully!")
            return RAGSystem(vector_store)
    except Exception as e:
        print(f"No existing vector store found or error loading it: {e}")
    
    # If loading failed or force_reload is True, create new vector store
    print("Creating new vector store...")
    print("1. Initializing document processor...")
    doc_processor = DocumentProcessor(source_dir=documents_dir)
    
    print("2. Loading documents...")
    documents = doc_processor.load_documents()
    if not documents:
        raise ValueError("No documents found in the specified directory!")
    
    print(f"Found {len(documents)} documents")
    
    print("3. Processing and chunking documents...")
    text_processor = TextProcessor()
    processed_docs = text_processor.process_documents(documents)
    
    print("4. Creating vector store...")
    vector_store = vector_store_manager.create_vector_store(processed_docs)
    
    # Save the update timestamp after successful creation
    vector_store_manager.save_last_update_time()
    
    print("5. Initializing RAG system...")
    return RAGSystem(vector_store)

# Create a global RAG system instance
_rag_system = None

def query_rag(question: str, include_sources: bool = False, force_reload: bool = False) -> str:
    """
    Query the RAG system with a single question.
    """
    try:
        global _rag_system
        
        # Check if documents have changed
        vector_store_manager = VectorStoreManager()
        documents_changed = vector_store_manager.needs_update("./documents")  # or your documents path
        
        if _rag_system is None:
            logger.info("RAG system not initialized - creating new instance")
        elif documents_changed:
            logger.info("Documents have changed - rebuilding vector store")
        elif force_reload:
            logger.info("Force reload requested - rebuilding vector store")
            
        # Initialize the RAG system if needed
        if _rag_system is None or documents_changed or force_reload:
            _rag_system = setup_rag_system(force_reload=force_reload)
        else:
            logger.info("Using existing RAG system")
        
        # Get response
        response = _rag_system.ask_question(question)
        
        # Format the response
        formatted_response = response["answer"]
        
        # Add sources if requested
        if include_sources:
            formatted_response += "\n\nSources:"
            for source in response["sources"]:
                formatted_response += f"\nFrom {source['source']}:\n{source['content']}"
        
        return formatted_response
            
    except Exception as e:
        logger.error(f"Error in query_rag: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the system
        answer = query_rag(
            "What suspicious activities do you see in the logs?",
            include_sources=True
        )
        print("Answer:", answer)
        
        # Follow-up question
        answer = query_rag(
            "Can you provide more details about any specific IP addresses?",
            include_sources=True
        )
        print("Answer:", answer)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")