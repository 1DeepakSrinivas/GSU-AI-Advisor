"""
Pinecone Database Setup for AI Advisor
Creates index and uploads vectorized data
"""

import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: Pinecone not installed. Install with: pip install pinecone")

class PineconeManager:
    
    def __init__(self):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Please install: pip install pinecone")
        
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")  # Default environment
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ai-advisor-index")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
    
    def create_index(self, dimension: int = 3072, metric: str = "cosine"):
        """Create a new Pinecone index"""
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return True
            
            # Create new index with serverless spec
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.environment
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while self.index_name not in [index.name for index in self.pc.list_indexes()]:
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            print(f"Index '{self.index_name}' created successfully!")
            return True
            
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def connect_to_index(self):
        """Connect to existing index"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"Index '{self.index_name}' does not exist. Create it first.")
                return False
            
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to index '{self.index_name}'")
            return True
            
        except Exception as e:
            print(f"Error connecting to index: {e}")
            return False
    
    def upload_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100):
        """Upload vectors to Pinecone in batches"""
        if not self.index:
            print("No index connected. Create or connect to an index first.")
            return False
        
        try:
            total_vectors = len(vectors)
            print(f"Uploading {total_vectors} vectors in batches of {batch_size}...")
            
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                
                # Format for Pinecone upload
                formatted_vectors = []
                for vector in batch:
                    formatted_vectors.append({
                        'id': vector['id'],
                        'values': vector['values'],
                        'metadata': vector['metadata']
                    })
                
                # Upload batch
                self.index.upsert(vectors=formatted_vectors)
                print(f"Uploaded batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1}")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
            
            print("All vectors uploaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error uploading vectors: {e}")
            return False
    
    def query_index(self, query_vector: List[float], top_k: int = 5, include_metadata: bool = True):
        """Query the index for similar vectors"""
        if not self.index:
            print("No index connected. Create or connect to an index first.")
            return None
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata
            )
            return results
            
        except Exception as e:
            print(f"Error querying index: {e}")
            return None
    
    def get_index_stats(self):
        """Get index statistics"""
        if not self.index:
            print("No index connected. Create or connect to an index first.")
            return None
        
        try:
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return None
    
    def delete_index(self):
        """Delete the index (use with caution!)"""
        try:
            confirm = input(f"Are you sure you want to delete index '{self.index_name}'? (yes/no): ")
            if confirm.lower() == 'yes':
                self.pc.delete_index(self.index_name)
                print(f"Index '{self.index_name}' deleted successfully!")
                self.index = None
                return True
            else:
                print("Index deletion cancelled.")
                return False
                
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False

    def create_rag_chain(self, retriever, system_prompt=None):
        """Create a RAG chain using a retriever and an LLM with optional system prompt"""
        try:
            # Initialize the LLM
            llm = ChatOpenAI()

            # Set default system prompt if none provided
            if system_prompt is None:
                system_prompt = """You are an AI advisor assistant. Use the provided context to answer questions accurately and helpfully. 
                If the answer cannot be found in the context, say so clearly. 
                Provide detailed, well-structured responses based on the available information."""

            # Create the RetrievalQA chain with custom prompt
            prompt_template = f"""{system_prompt}

Context: {{context}}

Question: {{question}}

Answer:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            # Return the callable chain
            return rag_chain

        except Exception as e:
            print(f"Error creating RAG chain: {e}")
            return None

def load_scraped_data(filename: str = "app/scraped_data.json") -> List[Dict[str, Any]]:
    """Load scraped data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to Pinecone format
        vectors = []
        for item in data:
            vector = {
                'id': item['id'],
                'values': item['embedding'],
                'metadata': {
                    'source_url': item['source_url'],
                    'title': item['title'],
                    'content': item['content'][:1000],  # Limit metadata size
                    'chunk_index': item['chunk_index'],
                    'total_chunks': item['total_chunks'],
                    'scraped_at': item['scraped_at']
                }
            }
            vectors.append(vector)
        
        return vectors
        
    except Exception as e:
        print(f"Error loading scraped data: {e}")
        return []

def main():
    """Main setup function"""
    if not PINECONE_AVAILABLE:
        print("Pinecone is not installed. Please run: pip install pinecone")
        return
    
    try:
        # Initialize Pinecone manager
        manager = PineconeManager()
        
        # Create or connect to index (using dimension 3072 for text-embedding-3-large)
        success = manager.create_index(dimension=3072)
        if not success:
            success = manager.connect_to_index()
        
        if not success:
            print("Failed to create or connect to index")
            return
        
        # Load scraped data
        vectors = load_scraped_data("app/scraped_data.json")
        
        if not vectors:
            print("No scraped data found. Run scraper.py first to generate data.")
            return
        
        # Upload vectors
        success = manager.upload_vectors(vectors)
        
        if success:
            # Show index stats
            stats = manager.get_index_stats()
            if stats:
                print(f"\nIndex Statistics:")
                print(f"Total vectors: {stats.get('total_vector_count', 'Unknown')}")
                print(f"Index fullness: {stats.get('index_fullness', 'Unknown')}")
        
        print("\nPinecone setup complete!")
        
    except Exception as e:
        print(f"Error in main setup: {e}")

if __name__ == "__main__":
    main() 