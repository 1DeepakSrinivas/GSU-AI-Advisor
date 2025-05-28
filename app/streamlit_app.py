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
from datetime import datetime

# Initialize components
manager = PineconeManager()
pdf_processor = PDFProcessor()
batch_processor = BatchProcessor()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Streamlit UI
st.title("GSU-AI-Advisor")
st.markdown("An advisor for all things GSU")
st.info("Information is accurate as of the 2020-21 Academic catalog")

# Initialize knowledge base
with st.spinner("Initializing knowledge base..."):
    batch_processor.ensure_knowledge_base_ready()

# Connect to existing index and set up retriever once
if st.session_state.retriever is None:
    if manager.connect_to_index():
        st.success(f"Connected to Pinecone index: {manager.index_name}")
        
        # Get index statistics
        stats = manager.get_index_stats()
        if stats:
            st.info(f"Index contains {stats.get('total_vector_count', 'Unknown')} vectors")
        
        # Set up retriever and RAG chain
        with st.spinner("Setting up retrieval system..."):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                retriever_setup = RetrieverSetup(manager.index_name)
                st.session_state.retriever = retriever_setup.setup_retriever()
            
            if st.session_state.retriever:
                st.success("Retrieval system ready!")
            else:
                st.error("Failed to set up retrieval system")
                st.stop()
    else:
        st.error("Failed to connect to Pinecone index. Please ensure the index exists and credentials are correct.")
        st.stop()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # System prompt configuration
    system_prompt = st.text_area(
        "System Prompt:",
        value="You are an AI Academic Advisor assistant for Georgia State University. Use the provided context and conversation history to answer questions accurately and helpfully. If the answer cannot be found in the context, say so clearly. Provide detailed, well-structured responses based on the available information. Include course codes, prerequisites, co-requisites, and all other necessary information. Build on previous conversation context when relevant.",
        height=150,
        help="Customize how the AI responds to your questions"
    )
    
    # Chat controls
    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()
    
    # Information
    st.header("Information")
    st.info("""
    This chat interface allows for:
    - Follow-up questions
    - Clarifications on previous answers
    - Building context over multiple exchanges
    - Progressive refinement of responses
    
    **Tips:**
    - Ask specific questions about courses
    - Reference previous answers for clarification
    - Build on topics gradually for better responses
    """)

# Chat interface
st.header("Chat with GSU Advisor")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Source Documents"):
                for j, source in enumerate(message["sources"]):
                    st.write(f"**Source {j+1}:**")
                    content = source.get("content", "")
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                    else:
                        st.write(content)
                    
                    if source.get("metadata"):
                        st.write(f"**Metadata:** {source['metadata']}")
                    st.write("---")

# Chat input
user_input = st.chat_input("Ask a question about GSU...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create or update RAG chain with current system prompt
                if st.session_state.rag_chain is None or system_prompt:
                    st.session_state.rag_chain = manager.create_rag_chain(
                        st.session_state.retriever, 
                        system_prompt
                    )
                
                if st.session_state.rag_chain:
                    # Build context from chat history
                    conversation_context = ""
                    if len(st.session_state.messages) > 1:
                        recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges
                        for msg in recent_messages[:-1]:  # Exclude current message
                            conversation_context += f"{msg['role'].title()}: {msg['content']}\n"
                    
                    # Create enhanced query with conversation context
                    enhanced_query = user_input
                    if conversation_context:
                        enhanced_query = f"Previous conversation:\n{conversation_context}\nCurrent question: {user_input}"
                    
                    # Query the RAG chain
                    try:
                        response = st.session_state.rag_chain.invoke({"query": enhanced_query})
                    except Exception as e:
                        try:
                            response = st.session_state.rag_chain.invoke({"question": enhanced_query})
                        except Exception as e2:
                            response = st.session_state.rag_chain.invoke({"input": enhanced_query})
                    
                    # Extract and display response
                    if isinstance(response, dict):
                        if "result" in response:
                            answer = response["result"]
                        elif "answer" in response:
                            answer = response["answer"]
                        else:
                            answer = str(response)
                        
                        st.write(answer)
                        
                        # Prepare sources for storage
                        sources = []
                        if "source_documents" in response and response["source_documents"]:
                            for doc in response["source_documents"]:
                                sources.append({
                                    "content": doc.page_content,
                                    "metadata": getattr(doc, 'metadata', {})
                                })
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Show sources in expandable section
                        if sources:
                            with st.expander("Source Documents"):
                                for i, source in enumerate(sources):
                                    st.write(f"**Source {i+1}:**")
                                    content = source.get("content", "")
                                    if len(content) > 500:
                                        st.write(content[:500] + "...")
                                    else:
                                        st.write(content)
                                    
                                    if source.get("metadata"):
                                        st.write(f"**Metadata:** {source['metadata']}")
                                    st.write("---")
                    else:
                        answer = str(response)
                        st.write(answer)
                        
                        # Add simple response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                else:
                    st.error("Failed to create RAG chain.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# Suggested follow-up questions based on chat context
if st.session_state.messages:
    st.subheader("Suggested Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Tell me more about prerequisites"):
            st.session_state.user_input = "Can you tell me more about the prerequisites for these courses?"
            st.rerun()
    
    with col2:
        if st.button("What about related courses?"):
            st.session_state.user_input = "What other related courses should I consider?"
            st.rerun()

# Display conversation summary in sidebar
if st.session_state.messages:
    with st.sidebar:
        st.header("Conversation Summary")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        if st.session_state.messages:
            st.write(f"**Started:** {st.session_state.messages[0]['timestamp'][:16]}")
            last_msg = st.session_state.messages[-1]
            st.write(f"**Last:** {last_msg['timestamp'][:16]}")
        
        # Export conversation option
        if st.button("Export Conversation"):
            conversation_text = ""
            for msg in st.session_state.messages:
                conversation_text += f"**{msg['role'].title()}:** {msg['content']}\n\n"
            
            st.download_button(
                label="Download Chat History",
                data=conversation_text,
                file_name=f"gsu_advisor_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            ) 