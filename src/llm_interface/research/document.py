"""
Document processing for retrieval-augmented generation.

This module provides tools for loading, processing, and chunking documents
for use in the retrieval system.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

from llm_interface.config import Config


class Document:
    """
    A document or chunk of text with metadata.
    
    Represents a document or text chunk along with associated metadata
    such as source, title, etc.
    """
    
    def __init__(self, 
                 text: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 doc_id: Optional[str] = None):
        """
        Initialize a document.
        
        Args:
            text: The document text
            metadata: Optional document metadata
            doc_id: Optional document identifier
        """
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id={self.doc_id}, text={self.text[:50]}...)"


class DocumentProcessor:
    """
    Processor for preparing documents for embedding and retrieval.
    
    This class provides methods for loading, cleaning, and chunking documents.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the document processor.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        
        # Default chunk settings from config
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for better chunking and embedding.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Convert tabs to spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure paragraphs are properly separated
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, 
                  text: str, 
                  chunk_size: Optional[int] = None,
                  chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # Clean the text first
        text = self.clean_text(text)
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a natural break point (paragraph or sentence)
            natural_end = self._find_natural_break(text, end)
            chunks.append(text[start:natural_end])
            
            # Move start position accounting for overlap
            start = natural_end - chunk_overlap
            
            # Make sure we're making forward progress
            if start >= natural_end:
                start = natural_end
        
        return chunks
    
    def _find_natural_break(self, text: str, position: int) -> int:
        """
        Find a natural break point near the position.
        
        Tries to find paragraph breaks, then sentence breaks, then word breaks.
        
        Args:
            text: The text to analyze
            position: The target position
            
        Returns:
            Position of the natural break
        """
        # Look for paragraph break
        paragraph_match = re.search(r'\n\n', text[position-100:position+100])
        if paragraph_match:
            return position - 100 + paragraph_match.start()
        
        # Look for sentence break (period followed by space)
        sentence_match = re.search(r'\.[ \n]', text[position-50:position+50])
        if sentence_match:
            return position - 50 + sentence_match.start() + 1
        
        # Look for word break (space)
        word_match = re.search(r'\s', text[position-20:position+20])
        if word_match:
            return position - 20 + word_match.start()
        
        # If no natural break found, just use the position
        return position
    
    def process_document(self, 
                        text: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a document into chunks with metadata.
        
        Args:
            text: The document text
            metadata: Optional document metadata
            
        Returns:
            List of Document objects
        """
        metadata = metadata or {}
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Create a copy of metadata for each chunk and add chunk info
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
            
            doc_id = metadata.get("doc_id")
            if doc_id:
                chunk_doc_id = f"{doc_id}_chunk_{i}"
            else:
                chunk_doc_id = f"chunk_{i}"
            
            documents.append(Document(
                text=chunk,
                metadata=chunk_metadata,
                doc_id=chunk_doc_id
            ))
        
        return documents
    
    def process_text_from_web(self, 
                             text: str, 
                             url: str, 
                             title: Optional[str] = None) -> List[Document]:
        """
        Process text from a web page.
        
        Args:
            text: The web page text
            url: The URL of the web page
            title: Optional web page title
            
        Returns:
            List of Document objects
        """
        metadata = {
            "source": "web",
            "url": url,
            "title": title or url,
            "doc_id": self._url_to_doc_id(url)
        }
        
        return self.process_document(text, metadata)
    
    def _url_to_doc_id(self, url: str) -> str:
        """
        Convert a URL to a document ID.
        
        Args:
            url: The URL
            
        Returns:
            A document ID based on the URL
        """
        # Remove protocol and common prefixes
        doc_id = re.sub(r'^https?://(www\.)?', '', url)
        
        # Replace non-alphanumeric characters with underscores
        doc_id = re.sub(r'[^a-zA-Z0-9]', '_', doc_id)
        
        # Limit length
        if len(doc_id) > 100:
            doc_id = doc_id[:100]
        
        return doc_id
