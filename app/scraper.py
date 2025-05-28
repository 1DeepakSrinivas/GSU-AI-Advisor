import os
import requests
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
import json
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class OpenAIEmbeddings:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                embeddings.append(response.data[0].embedding)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embed_documents([text])[0]

class WebScraper:
    """Web scraper with chunking and embedding capabilities"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings()
        self.scraped_data = []
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL and return structured data"""
        try:
            print(f"Scraping: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Get main content (try to find main content areas)
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.post-content', '.entry-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text()
                    break
            
            # If no main content found, get body text
            if not content_text:
                body = soup.find('body')
                content_text = body.get_text() if body else soup.get_text()
            
            # Clean up text
            content_text = ' '.join(content_text.split())
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'scraped_at': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                'url': url,
                'title': "",
                'content': "",
                'scraped_at': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs and process them into chunks with embeddings"""
        all_chunks = []
        
        for url in urls:
            scraped_data = self.scrape_url(url)
            
            if scraped_data['success'] and scraped_data['content']:
                chunks = self.text_splitter.split_text(scraped_data['content'])
                print(f"Created {len(chunks)} chunks from {url}")
                
                # Generate embeddings for chunks
                print("Generating embeddings...")
                embeddings = self.embeddings.embed_documents(chunks)
                
                # Create chunk objects
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_data = {
                        'id': f"{url}_{i}",
                        'source_url': url,
                        'title': scraped_data['title'],
                        'content': chunk,
                        'embedding': embedding,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'scraped_at': scraped_data['scraped_at']
                    }
                    all_chunks.append(chunk_data)
            time.sleep(0.1)
        
        self.scraped_data = all_chunks
        return all_chunks
    
    def save_to_json(self, filename: str = "scraped_data.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    
    def get_pinecone_vectors(self) -> List[Dict[str, Any]]:
        """Format data for Pinecone upload"""
        vectors = []
        for chunk in self.scraped_data:
            vector = {
                'id': chunk['id'],
                'values': chunk['embedding'],
                'metadata': {
                    'source_url': chunk['source_url'],
                    'title': chunk['title'],
                    'content': chunk['content'][:1000],  # Limit metadata size
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'scraped_at': chunk['scraped_at']
                }
            }
            vectors.append(vector)
        return vectors

def main():
    urls = [
        # Add your URLs here
        "https://catalogs.gsu.edu/mime/media/6/8384/2020-2021_Undergraduate_Catalog_Bachelor-Level.pdf",
    ]
    
    if not urls:
        print("Please add URLs to the 'urls' list in the main() function")
        return
    
    # Initialize scraper
    scraper = WebScraper(chunk_size=1000, chunk_overlap=200)
    
    # Process URLs
    chunks = scraper.process_urls(urls)
    print(f"\nProcessed {len(chunks)} total chunks from {len(urls)} URLs")
    
    # Save to JSON
    scraper.save_to_json("app/scraped_data.json")
    
    # Get vectors for Pinecone
    vectors = scraper.get_pinecone_vectors()
    print(f"Ready to upload {len(vectors)} vectors to Pinecone")
    
    return scraper, vectors

if __name__ == "__main__":
    main() 
    