"""
PDF Processor for AI Advisor
Downloads PDFs, extracts text, creates embeddings, and pushes directly to Pinecone
"""

import os
import requests
import pdfplumber
import uuid
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.pinecone_setup import PineconeManager

# Load environment variables
load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.pinecone_manager = PineconeManager()
        
    def download_pdf(self, url: str, temp_filename: str = "temp_document.pdf") -> str:
        """Download PDF from URL to temporary file"""
        try:
            print(f"Downloading PDF from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"PDF downloaded successfully: {temp_filename}")
            return temp_filename
            
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF file"""
        try:
            print(f"Extracting text from: {pdf_path}")
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text.strip())
                            print(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            print(f"Successfully extracted text from {len(text_content)} pages")
            return text_content
            
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []
    
    def chunk_text(self, texts: List[str]) -> List[str]:
        """Split texts into smaller chunks for embedding"""
        try:
            print("Chunking text for optimal embedding...")
            all_chunks = []
            
            for text in texts:
                chunks = self.text_splitter.split_text(text)
                all_chunks.extend(chunks)
            
            print(f"Created {len(all_chunks)} text chunks")
            return all_chunks
            
        except Exception as e:
            print(f"Error chunking text: {e}")
            return []
    
    def create_embeddings(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Create embeddings for text chunks"""
        try:
            print("Creating embeddings...")
            embedded_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create embedding
                    embedding = self.embeddings.embed_query(chunk)
                    
                    # Create document with metadata
                    doc = {
                        'id': str(uuid.uuid4()),
                        'values': embedding,
                        'metadata': {
                            'content': chunk,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'processed_at': datetime.now().isoformat(),
                            'source_type': 'pdf'
                        }
                    }
                    embedded_chunks.append(doc)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Created embeddings for {i + 1}/{len(chunks)} chunks")
                        
                except Exception as e:
                    print(f"Error creating embedding for chunk {i}: {e}")
                    continue
            
            print(f"Successfully created {len(embedded_chunks)} embeddings")
            return embedded_chunks
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []
    
    def push_to_pinecone(self, embedded_chunks: List[Dict[str, Any]], url: str, title: str = None) -> bool:
        """Push embeddings directly to Pinecone"""
        try:
            print("Connecting to Pinecone...")
            
            # Connect to existing index or create new one
            success = self.pinecone_manager.connect_to_index()
            if not success:
                print("Creating new Pinecone index...")
                success = self.pinecone_manager.create_index(dimension=3072)
            
            if not success:
                print("Failed to connect to Pinecone")
                return False
            
            # Add URL and title to metadata
            for chunk in embedded_chunks:
                chunk['metadata']['source_url'] = url
                chunk['metadata']['title'] = title or 'PDF Document'
            
            # Upload to Pinecone
            print(f"Uploading {len(embedded_chunks)} embeddings to Pinecone...")
            success = self.pinecone_manager.upload_vectors(embedded_chunks)
            
            if success:
                print("Successfully uploaded embeddings to Pinecone!")
                
                # Show updated stats
                stats = self.pinecone_manager.get_index_stats()
                if stats:
                    print(f"Updated index stats: {stats.get('total_vector_count', 'Unknown')} total vectors")
                
                return True
            else:
                print("Failed to upload embeddings to Pinecone")
                return False
                
        except Exception as e:
            print(f"Error pushing to Pinecone: {e}")
            return False
    
    def process_pdf_url(self, url: str, title: str = None) -> tuple[bool, int]:
        """Complete pipeline: download PDF, extract text, create embeddings, push to Pinecone"""
        try:
            print(f"\n{'='*60}")
            print(f"Processing PDF: {title or 'PDF Document'}")
            print(f"URL: {url}")
            print(f"{'='*60}")
            
            # Step 1: Download PDF
            temp_pdf = self.download_pdf(url)
            if not temp_pdf:
                return False, 0
            
            try:
                # Step 2: Extract text
                texts = self.extract_text_from_pdf(temp_pdf)
                if not texts:
                    print("No text extracted from PDF")
                    return False, 0
                
                # Step 3: Chunk text
                chunks = self.chunk_text(texts)
                if not chunks:
                    print("No text chunks created")
                    return False, 0
                
                # Step 4: Create embeddings
                embedded_chunks = self.create_embeddings(chunks)
                if not embedded_chunks:
                    print("No embeddings created")
                    return False, 0
                
                # Step 5: Push to Pinecone
                success = self.push_to_pinecone(embedded_chunks, url, title)
                
                chunks_count = len(embedded_chunks)
                
                if success:
                    print(f"\nSuccessfully processed PDF: {chunks_count} chunks added to knowledge base")
                else:
                    print(f"\nFailed to process PDF")
                
                return success, chunks_count
                
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_pdf)
                    print(f"Cleaned up temporary file: {temp_pdf}")
                except:
                    pass
                    
        except Exception as e:
            print(f"Error in PDF processing pipeline: {e}")
            return False, 0

def main():
    """Example usage"""
    processor = PDFProcessor()
    
    # Example: Process the GSU catalog
    gsu_catalog_url = "https://catalogs.gsu.edu/mime/media/6/8384/2020-2021_Undergraduate_Catalog_Bachelor-Level.pdf"
    
    success, chunks_count = processor.process_pdf_url(
        url=gsu_catalog_url,
        title="GSU 2020-2021 Undergraduate Catalog"
    )
    
    if success:
        print("\nPDF processing completed successfully!")
    else:
        print("\nPDF processing failed!")

if __name__ == "__main__":
    main() 