"""
Vector retrieval for RAG.

This module provides tools for creating, storing, and retrieving
vector embeddings for document retrieval.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from llm_interface.config import Config
from llm_interface.research.document import Document


class SimpleVectorStore:
    """
    A simple vector store for document embeddings.
    
    This class provides methods for storing and retrieving document
    embeddings using a simple file-based approach.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.embeddings_dir = self.config["embeddings_dir"]
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize empty store
        self.embeddings = {}
        self.documents = {}
        self.faiss_index = None
        
        # Default to L2 distance
        self.metric = "l2"
    
    def add_embeddings(self, 
                       documents: List[Document], 
                       embeddings: List[List[float]],
                       collection_name: str = "default",
                       debug: bool = False) -> None:
        """
        Add document embeddings to the store.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors (one per document)
            collection_name: Name of the collection
            debug: Whether to print debug information
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        # Create collection directory if it doesn't exist
        collection_dir = os.path.join(self.embeddings_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Add documents and embeddings to memory store
        for doc, emb in zip(documents, embeddings):
            doc_id = doc.doc_id or str(len(self.documents))
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = emb
        
        # Save collection to disk
        self._save_collection(collection_name)
        
        # Rebuild FAISS index if available
        if FAISS_AVAILABLE:
            self._build_faiss_index()
            
        if debug:
            print(f"DEBUG - Added {len(documents)} documents to vector store collection '{collection_name}'")
    
    def _save_collection(self, collection_name: str) -> None:
        """
        Save a collection to disk.
        
        Args:
            collection_name: Name of the collection
        """
        collection_dir = os.path.join(self.embeddings_dir, collection_name)
        
        # Save documents
        documents_data = {}
        for doc_id, doc in self.documents.items():
            documents_data[doc_id] = {
                "text": doc.text,
                "metadata": doc.metadata,
                "doc_id": doc.doc_id
            }
        
        with open(os.path.join(collection_dir, "documents.json"), 'w') as f:
            json.dump(documents_data, f)
        
        # Save embeddings
        embeddings_data = {}
        for doc_id, emb in self.embeddings.items():
            embeddings_data[doc_id] = emb
        
        np.save(os.path.join(collection_dir, "embeddings.npy"), embeddings_data)
    
    def load_collection(self, collection_name: str = "default", debug: bool = False) -> bool:
        """
        Load a collection from disk.
        
        Args:
            collection_name: Name of the collection
            debug: Whether to print debug information
            
        Returns:
            True if the collection was loaded, False otherwise
        """
        collection_dir = os.path.join(self.embeddings_dir, collection_name)
        
        documents_path = os.path.join(collection_dir, "documents.json")
        embeddings_path = os.path.join(collection_dir, "embeddings.npy")
        
        if not os.path.exists(documents_path) or not os.path.exists(embeddings_path):
            if debug:
                print(f"DEBUG - Collection '{collection_name}' not found")
            return False
        
        # Load documents
        with open(documents_path, 'r') as f:
            documents_data = json.load(f)
        
        self.documents = {}
        for doc_id, doc_data in documents_data.items():
            self.documents[doc_id] = Document(
                text=doc_data["text"],
                metadata=doc_data["metadata"],
                doc_id=doc_data["doc_id"]
            )
        
        # Load embeddings
        self.embeddings = np.load(embeddings_path, allow_pickle=True).item()
        
        # Rebuild FAISS index if available
        if FAISS_AVAILABLE:
            self._build_faiss_index()
        
        if debug:
            print(f"DEBUG - Loaded collection '{collection_name}' with {len(self.documents)} documents")
        
        return True
    
    def _build_faiss_index(self) -> None:
        """Build a FAISS index for fast vector similarity search."""
        if not FAISS_AVAILABLE or not self.embeddings:
            return
        
        # Convert embeddings to numpy array
        doc_ids = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[doc_id] for doc_id in doc_ids])
        dim = embeddings_array.shape[1]
        
        # Create index based on metric
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.faiss_index = faiss.IndexFlatIP(dim)
        else:
            # L2 distance
            self.faiss_index = faiss.IndexFlatL2(dim)
        
        self.faiss_index.add(embeddings_array)
        self.faiss_doc_ids = doc_ids
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         k: int = 5,
                         debug: bool = False) -> List[Tuple[Document, float]]:
        """
        Find similar documents by embedding.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            debug: Whether to print debug information
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.embeddings:
            if debug:
                print("DEBUG - No embeddings available for search")
            return []
        
        k = min(k, len(self.embeddings))
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Use FAISS for fast search
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            if self.metric == "cosine":
                # Normalize query for cosine similarity
                faiss.normalize_L2(query_embedding_array)
                similarities, indices = self.faiss_index.search(query_embedding_array, k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.faiss_doc_ids):
                        doc_id = self.faiss_doc_ids[idx]
                        score = similarities[0][i]  # Higher is better for cosine
                        results.append((self.documents[doc_id], score))
                
                if debug:
                    print(f"DEBUG - Found {len(results)} documents using FAISS cosine similarity search")
                
                return results
            else:
                # L2 distance (lower is better)
                distances, indices = self.faiss_index.search(query_embedding_array, k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.faiss_doc_ids):
                        doc_id = self.faiss_doc_ids[idx]
                        score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
                        results.append((self.documents[doc_id], score))
                
                if debug:
                    print(f"DEBUG - Found {len(results)} documents using FAISS L2 distance search")
                
                return results
        else:
            # Fallback to numpy
            if debug:
                print("DEBUG - FAISS not available, using numpy for similarity search")
                
            scores = []
            for doc_id, embedding in self.embeddings.items():
                if self.metric == "cosine":
                    # Cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                else:
                    # L2 distance
                    distance = np.linalg.norm(np.array(query_embedding) - np.array(embedding))
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                scores.append((doc_id, similarity))
            
            # Sort by similarity (highest first)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            results = [(self.documents[doc_id], score) for doc_id, score in scores[:k]]
            
            if debug:
                print(f"DEBUG - Found {len(results)} documents using numpy similarity search")
                
            return results


class Embedder:
    """
    Text embedding interface.
    
    This class provides methods for creating vector embeddings from text.
    Uses sentence-transformers if available, with a simple fallback.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the embedder.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.model_name = self.config["embeddings_model"]
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.use_sentence_transformers = True
        except ImportError:
            print("WARNING: sentence-transformers not available, using simple fallback embedding")
            self.use_sentence_transformers = False
    
    def embed_texts(self, texts: List[str], debug: bool = False) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            debug: Whether to print debug information
            
        Returns:
            List of embedding vectors
        """
        if self.use_sentence_transformers:
            # Use sentence-transformers for high-quality embeddings
            if debug:
                print(f"DEBUG - Creating {len(texts)} embeddings using sentence-transformers model '{self.model_name}'")
                
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        else:
            # Simple fallback using word frequency
            if debug:
                print(f"DEBUG - Creating {len(texts)} embeddings using simple fallback method")
                
            return [self._simple_embed(text) for text in texts]
    
    def embed_text(self, text: str, debug: bool = False) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: Text string
            debug: Whether to print debug information
            
        Returns:
            Embedding vector
        """
        return self.embed_texts([text], debug=debug)[0]
    
    def _simple_embed(self, text: str, dim: int = 100) -> List[float]:
        """
        Create a simple embedding based on word frequency.
        
        This is a very basic fallback and should not be used for production.
        
        Args:
            text: Text to embed
            dim: Dimensionality of the embedding
            
        Returns:
            Embedding vector
        """
        import hashlib
        
        # Normalize text
        text = text.lower()
        words = text.split()
        
        # Count words
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create embedding
        embedding = [0.0] * dim
        for word, count in word_counts.items():
            # Hash the word to get indices
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % dim
            
            # Add count to embedding
            embedding[idx] += count
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [e / norm for e in embedding]
        
        return embedding


class RetrieverRag:
    """
    Retrieval-Augmented Generation.
    
    This class combines the document processor, embedder, and vector store
    to provide RAG capabilities for the LLM.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.document_processor = None
        self.embedder = None
        self.vector_store = None
    
    def _lazy_init(self) -> None:
        """Lazy initialization of components to avoid unnecessary imports."""
        if self.document_processor is None:
            from llm_interface.research.document import DocumentProcessor
            self.document_processor = DocumentProcessor(self.config)
        
        if self.embedder is None:
            self.embedder = Embedder(self.config)
        
        if self.vector_store is None:
            self.vector_store = SimpleVectorStore(self.config)
            # Try to load default collection
            self.vector_store.load_collection()
    
    def add_web_content(self, 
                       text: str, 
                       url: str, 
                       title: Optional[str] = None,
                       collection_name: str = "default",
                       debug: bool = False) -> None:
        """
        Add web content to the RAG system.
        
        Args:
            text: The web page text
            url: The URL of the web page
            title: Optional web page title
            collection_name: Name of the collection
            debug: Whether to print debug information
        """
        self._lazy_init()
        
        # Process text into documents
        documents = self.document_processor.process_text_from_web(text, url, title)
        
        if not documents:
            if debug:
                print(f"DEBUG - No documents created from {url}")
            return
            
        # Extract text for embedding
        texts = [doc.text for doc in documents]
        
        # Create embeddings
        embeddings = self.embedder.embed_texts(texts, debug=debug)
        
        # Add to vector store
        self.vector_store.add_embeddings(documents, embeddings, collection_name, debug=debug)
        
        if debug:
            print(f"DEBUG - Added {len(documents)} documents from {url} to vector store")
    
    def add_web_research(self, research_data: Dict[str, Any], debug: bool = False) -> None:
        """
        Add web research results to the RAG system.
        
        Args:
            research_data: Research data from the WebResearcher
            debug: Whether to print debug information
        """
        self._lazy_init()
        
        # Process content from research results
        for i, item in enumerate(research_data.get("content", [])):
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            
            if not content or not url:
                continue
                
            if debug:
                print(f"DEBUG - Adding web content from {url} to vector store")
                
            # Add to RAG
            self.add_web_content(content, url, title, debug=debug)
    
    def query(self, query: str, k: int = 5, debug: bool = False) -> List[Document]:
        """
        Query the RAG system.
        
        Args:
            query: The query text
            k: Number of results to return
            debug: Whether to print debug information
            
        Returns:
            List of relevant documents
        """
        self._lazy_init()
        
        # Embed query
        query_embedding = self.embedder.embed_text(query, debug=debug)
        
        # Search for similar documents
        results = self.vector_store.similarity_search(query_embedding, k, debug=debug)
        
        documents = [doc for doc, score in results]
        
        if debug:
            print(f"DEBUG - Retrieved {len(documents)} documents for query: {query}")
            
        return documents
    
    def format_context_for_prompt(self, documents: List[Document], debug: bool = False) -> str:
        """
        Format retrieved documents for inclusion in an LLM prompt.
        
        Args:
            documents: List of retrieved documents
            debug: Whether to print debug information
            
        Returns:
            Formatted context text for inclusion in an LLM prompt
        """
        context_parts = ["RELEVANT INFORMATION FROM KNOWLEDGE BASE:"]
        
        for i, doc in enumerate(documents, 1):
            source_info = ""
            if doc.metadata.get("url"):
                source_info = f" (Source: {doc.metadata.get('url')})"
            
            context_parts.append(
                f"[DOCUMENT {i}]{source_info}\n"
                f"{doc.text}\n"
            )
        
        if debug:
            print(f"DEBUG - Formatted {len(documents)} documents for prompt")
            
        return "\n".join(context_parts)