import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

class RetrieverSetup:
    def __init__(self, index_name):
        self.index_name = index_name
        self.api_key = os.getenv("PINECONE_API_KEY")
        
    def setup_retriever(self):
        try:
            print(f"Setting up retriever for index: {self.index_name}")
            
            # Check if API key is available
            if not self.api_key:
                print("Error: PINECONE_API_KEY not found in environment variables")
                return None
            
            # Initialize OpenAI embeddings with the correct model for 3072 dimensions
            print("Initializing OpenAI embeddings...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            print(f"Creating vector store for index '{self.index_name}'...")
            
            # Create a Pinecone vector store from the existing index using the new API
            vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=embeddings,
                text_key="content"  # This is the key name used in your stored data
            )
            
            print("Vector store created successfully. Setting up retriever...")
            
            # Return the retriever
            retriever = vector_store.as_retriever()
            print("Retriever setup completed successfully!")
            return retriever

        except Exception as e:
            print(f"Error setting up retriever: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None 

    def test_retriever(self, query="economics classes"):
        """Test the retriever to see if it can find relevant documents"""
        try:
            retriever = self.setup_retriever()
            if not retriever:
                print("Failed to setup retriever")
                return False
                
            print(f"Testing retriever with query: '{query}'")
            
            # Test retrieval
            docs = retriever.get_relevant_documents(query)
            
            print(f"Found {len(docs)} documents")
            
            if docs:
                print("\n--- Sample Retrieved Document ---")
                print(f"Content preview: {docs[0].page_content[:200]}...")
                if hasattr(docs[0], 'metadata') and docs[0].metadata:
                    print(f"Metadata: {docs[0].metadata}")
                print("--- End Sample ---\n")
                return True
            else:
                print("No documents retrieved")
                return False
                
        except Exception as e:
            print(f"Error testing retriever: {e}")
            import traceback
            traceback.print_exc()
            return False 