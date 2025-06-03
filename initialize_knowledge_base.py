
from app.batch_processor import create_default_knowledge_base

def main():
    print("Initializing AI Advisor Knowledge Base...")
    print("This will process essential documents and create embeddings.")
    print("This may take several minutes depending on document size.\n")
    
    success = create_default_knowledge_base()
    
    if success:
        print("\nKnowledge base initialization completed successfully!")
        print("You can now run the Streamlit app to start asking questions.")
        print("\nTo start the app, run:")
        print("streamlit run app/streamlit_app.py")
    else:
        print("\nKnowledge base initialization failed!")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 