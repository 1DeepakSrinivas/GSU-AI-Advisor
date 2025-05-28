import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.pinecone_setup import PineconeManager
from app.retriever import RetrieverSetup
from app.pdf_processor import PDFProcessor
from app.batch_processor import BatchProcessor
import io
import contextlib

# Initialize components
manager = PineconeManager()
pdf_processor = PDFProcessor()
batch_processor = BatchProcessor()

# Streamlit UI
st.title("GSU-AI-Advisor")
st.markdown("An advisor for all things GSU")
st.info("Information is accurate as of the 2020-21 Academic catalog")

# Initialize knowledge base
with st.spinner("Initializing knowledge base..."):
    batch_processor.ensure_knowledge_base_ready()

# Connect to existing index
if manager.connect_to_index():
    st.success(f"Connected to Pinecone index: {manager.index_name}")
    
    # Get index statistics
    stats = manager.get_index_stats()
    if stats:
        st.info(f"Index contains {stats.get('total_vector_count', 'Unknown')} vectors")
else:
    st.error("Failed to connect to Pinecone index. Please ensure the index exists and credentials are correct.")
    st.stop()

# System prompt input
st.header("Assistant Configuration")
system_prompt = st.text_area(
    "System Prompt (Optional):",
    value="You are an AI Academic Advisor assistant. Use the provided context to answer questions accurately and helpfully. If the answer cannot be found in the context, say so clearly. Provide detailed, well-structured responses based on the available information. All answers must be relevant to Georgia State University. Provide course codes, their prerequisites and co-requisites, and all other necessary information along with the answer for the user to be aware of.",
    height=100,
    help="Customize how the AI responds to your questions"
)

# Question input
st.header("Ask a Question")
question = st.text_input("Enter your question:", placeholder="What economics courses are available for first-year students?")

if question:
    with st.spinner("Processing your question..."):
        try:
            # Capture output for debugging
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                # Set up retriever and RAG chain
                retriever_setup = RetrieverSetup(manager.index_name)
                retriever = retriever_setup.setup_retriever()
            
            # Get captured output
            output = f.getvalue()
            if output:
                with st.expander("Debug Information"):
                    st.text(output)
            
            if retriever:
                # Create RAG chain with custom system prompt
                rag_chain = manager.create_rag_chain(retriever, system_prompt)
                
                if rag_chain:
                    # Query the RAG chain
                    try:
                        print(f"Querying RAG chain with: {question}")
                        response = rag_chain.invoke({"query": question})
                        print(f"RAG chain response received: {type(response)}")
                    except Exception as e:
                        print(f"Error with 'query' format, trying 'question': {e}")
                        try:
                            response = rag_chain.invoke({"question": question})
                        except Exception as e2:
                            print(f"Error with 'question' format, trying direct call: {e2}")
                            response = rag_chain.invoke({"input": question})
                    
                    st.subheader("Response:")
                    
                    # Handle different response formats
                    if isinstance(response, dict):
                        if "result" in response:
                            st.write(response["result"])
                        elif "answer" in response:
                            st.write(response["answer"])
                        else:
                            st.write(str(response))
                        
                        # Show source documents if available
                        if "source_documents" in response and response["source_documents"]:
                            with st.expander("Source Documents"):
                                for i, doc in enumerate(response["source_documents"]):
                                    st.write(f"**Source {i+1}:**")
                                    # Display content preview
                                    content = doc.page_content
                                    if len(content) > 500:
                                        st.write(content[:500] + "...")
                                    else:
                                        st.write(content)
                                    
                                    # Display metadata if available
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.write(f"**Metadata:** {doc.metadata}")
                                    st.write("---")
                        else:
                            st.info("No source documents found for this query.")
                    else:
                        st.write(str(response))
                else:
                    st.error("Failed to create RAG chain.")
            else:
                st.error("Failed to set up retriever. Check the debug information above for details.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Display helpful information
st.sidebar.header("Information")
st.sidebar.info("""
This AI Advisor can:
- Answer questions about Georgia State University
- Provide information about courses, programs, and requirements
- Give details about prerequisites and co-requisites
- Help with academic planning and course selection

**How to use:**
1. Enter your question in the text input
2. Review the AI's response
3. Check source documents for verification
4. Customize the system prompt if needed
""")

st.sidebar.header("System Prompt")
st.sidebar.info("""
The system prompt guides how the AI responds. You can customize it to:
- Change the AI's personality or role
- Add specific instructions or requirements
- Focus on particular topics or domains
- Adjust the response style and format
""") 