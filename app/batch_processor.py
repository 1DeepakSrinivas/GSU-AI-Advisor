import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from app.pdf_processor import PDFProcessor
from app.pinecone_setup import PineconeManager

# Load environment variables
load_dotenv()

class BatchProcessor:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.pinecone_manager = PineconeManager()
        # Using Pinecone only
        
    def load_catalog(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.catalog_file):
                with open(self.catalog_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"documents": [], "last_updated": None, "total_processed": 0}
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return {"documents": [], "last_updated": None, "total_processed": 0}
    
    def save_catalog(self):
        try:
            self.document_catalog["last_updated"] = datetime.now().isoformat()
            with open(self.catalog_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_catalog, f, indent=2, ensure_ascii=False)
            print(f"Catalog updated: {self.catalog_file}")
        except Exception as e:
            print(f"Error saving catalog: {e}")
    
    def is_document_processed(self, url: str) -> bool:
        for doc in self.document_catalog["documents"]:
            if doc["url"] == url:
                return True
        return False
    
    def add_to_catalog(self, url: str, title: str, chunks_count: int, success: bool):
        doc_entry = {
            "url": url,
            "title": title,
            "processed_at": datetime.now().isoformat(),
            "chunks_count": chunks_count,
            "success": success,
            "document_id": f"doc_{len(self.document_catalog['documents']) + 1}"
        }
        
        self.document_catalog["documents"].append(doc_entry)
        if success:
            self.document_catalog["total_processed"] += 1
        self.save_catalog()
    
    def get_catalog_summary(self) -> Dict[str, Any]:
        total_docs = len(self.document_catalog["documents"])
        successful_docs = sum(1 for doc in self.document_catalog["documents"] if doc["success"])
        total_chunks = sum(doc["chunks_count"] for doc in self.document_catalog["documents"] if doc["success"])
        
        return {
            "total_documents": total_docs,
            "successful_documents": successful_docs,
            "failed_documents": total_docs - successful_docs,
            "total_chunks": total_chunks,
            "last_updated": self.document_catalog.get("last_updated")
        }
    
    def process_document_list(self, documents: List[Dict[str, str]], force_reprocess: bool = False) -> Dict[str, Any]:
        results = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "details": []
        }
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING: {len(documents)} documents")
        print(f"{'='*80}")
        
        for i, doc in enumerate(documents, 1):
            url = doc["url"]
            title = doc.get("title", f"Document {i}")
            
            print(f"\n[{i}/{len(documents)}] Processing: {title}")
            print(f"URL: {url}")
            
            # Check if already processed
            if not force_reprocess and self.is_document_processed(url):
                print(f"Skipping - already processed: {title}")
                results["skipped"] += 1
                results["details"].append({
                    "title": title,
                    "url": url,
                    "status": "skipped",
                    "reason": "already_processed"
                })
                continue
            
            # Process the document
            try:
                success, chunks_count = self.pdf_processor.process_pdf_url(url, title)
                
                if success:
                    print(f"Successfully processed: {title} ({chunks_count} chunks)")
                    results["processed"] += 1
                    results["details"].append({
                        "title": title,
                        "url": url,
                        "status": "success",
                        "chunks_count": chunks_count
                    })
                    self.add_to_catalog(url, title, chunks_count, True)
                else:
                    print(f"Failed to process: {title}")
                    results["failed"] += 1
                    results["details"].append({
                        "title": title,
                        "url": url,
                        "status": "failed",
                        "reason": "processing_error"
                    })
                    self.add_to_catalog(url, title, 0, False)
                    
            except Exception as e:
                print(f"Error processing {title}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "title": title,
                    "url": url,
                    "status": "failed",
                    "reason": str(e)
                })
                self.add_to_catalog(url, title, 0, False)
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Processed: {results['processed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Failed: {results['failed']}")
        print(f"{'='*80}")
        
        return results
    
    def ensure_knowledge_base_ready(self) -> bool:
        print("Checking Pinecone knowledge base...")
        
        # Connect to Pinecone index
        if not self.pinecone_manager.connect_to_index():
            print("Failed to connect to Pinecone index")
            return False
        
        # Check if Pinecone has content
        stats = self.pinecone_manager.get_index_stats()
        if stats and stats.get('total_vector_count', 0) > 0:
            vector_count = stats.get('total_vector_count', 0)
            print(f"Knowledge base ready - {vector_count} vectors available in Pinecone")
            return True
        else:
            print("Pinecone index is empty - no knowledge base content found")
            return False
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        return [doc for doc in self.document_catalog["documents"] if doc["success"]]
    
    def remove_document_from_catalog(self, url: str):
        self.document_catalog["documents"] = [
            doc for doc in self.document_catalog["documents"] 
            if doc["url"] != url
        ]
        self.save_catalog()
        print(f"Removed document from catalog: {url}")

def create_default_knowledge_base():
    processor = BatchProcessor()
    
    print("Checking knowledge base availability...")
    success = processor.ensure_knowledge_base_ready()
    
    if success:
        print("Knowledge base is ready!")
    else:
        print("No knowledge base content found in Pinecone")
    
    return success

def main():
    processor = BatchProcessor()
    
    # Example: Process multiple documents
    documents_to_process = [
        {
            "url": "https://catalogs.gsu.edu/mime/media/6/8384/2020-2021_Undergraduate_Catalog_Bachelor-Level.pdf",
            "title": "GSU 2020-2021 Undergraduate Catalog"
        }
        # Add more documents here
    ]
    
    # Process the documents
    results = processor.process_document_list(documents_to_process)
    
    # Show summary
    summary = processor.get_catalog_summary()
    print(f"\nFinal Summary:")
    print(f"Total documents in catalog: {summary['total_documents']}")
    print(f"Successfully processed: {summary['successful_documents']}")
    print(f"Total chunks in knowledge base: {summary['total_chunks']}")

if __name__ == "__main__":
    main() 