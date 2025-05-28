# GSU-AI-Advisor

An AI-powered academic advisor for Georgia State University students, providing information about courses, programs, and requirements based on the 2020-21 Academic catalog.

## Features

- **Interactive Chat Interface**: Ask questions about GSU courses, requirements, and programs
- **Document-Based Responses**: Answers are grounded in official GSU catalog information
- **Source Citation**: View the source documents used to generate each response
- **Real-time Processing**: Instant answers using Pinecone vector database
- **Customizable System Prompts**: Adjust the AI's response style and focus

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: LangChain for RAG (Retrieval-Augmented Generation)
- **Vector Database**: Pinecone for semantic search
- **Embeddings**: OpenAI text-embedding-3-large model
- **LLM**: OpenAI GPT-3.5-turbo
- **Document Processing**: PDF text extraction with pdfplumber

## Setup Instructions

### 1. Environment Setup

Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=gsu-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

- streamlit
- langchain
- langchain-openai
- langchain-pinecone
- langchain-community
- pinecone-client
- pdfplumber
- python-dotenv

### 3. Initialize Knowledge Base

The system will automatically check and use existing Pinecone vectors. If you need to set up from scratch:

```bash
python initialize_knowledge_base.py
```

### 4. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Start the Application**: Run the Streamlit command above
2. **Wait for Initialization**: The system checks Pinecone for existing knowledge base
3. **Ask Questions**: Enter questions about GSU in the text input
4. **Review Responses**: Get detailed answers with course codes and requirements
5. **Check Sources**: Expand "Source Documents" to verify information

### Example Questions

- "What economics courses are available for first-year students?"
- "What are the prerequisites for MATH 2211?"
- "Tell me about the Computer Science degree requirements"
- "What courses satisfy the core curriculum for Area F?"

## File Structure

```
GSU-AI-Advisor/
├── app/
│   ├── streamlit_app.py      # Main Streamlit application
│   ├── pinecone_setup.py     # Pinecone database management
│   ├── retriever.py          # Document retrieval setup
│   ├── pdf_processor.py      # PDF processing and embedding
│   └── batch_processor.py    # Batch document processing
├── initialize_knowledge_base.py  # Setup script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
└── README.md                # This file
```

## System Components

### Core Components

- **PineconeManager**: Handles vector database operations
- **PDFProcessor**: Downloads and processes PDF documents
- **BatchProcessor**: Manages document catalog and processing
- **RetrieverSetup**: Configures semantic search retrieval
- **Streamlit App**: User interface and interaction handling

### Data Flow

1. **Document Processing**: PDFs are downloaded, text extracted, and chunked
2. **Embedding Generation**: Text chunks are converted to vectors using OpenAI
3. **Vector Storage**: Embeddings are stored in Pinecone with metadata
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Response Generation**: Retrieved context is used by GPT-3.5 to generate answers

## Configuration

### System Prompt Customization

The default system prompt can be modified in the Streamlit interface:

```
You are an AI Academic Advisor assistant. Use the provided context to answer questions accurately and helpfully. If the answer cannot be found in the context, say so clearly. Provide detailed, well-structured responses based on the available information. All answers must be relevant to Georgia State University. Provide course codes, their prerequisites and co-requisites, and all other necessary information along with the answer for the user to be aware of.
```

### Vector Database Settings

- **Embedding Model**: text-embedding-3-large (3072 dimensions)
- **Chunk Size**: 1000 characters with 200 character overlap
- **Index Name**: gsu-ai (configurable via environment)

## Troubleshooting

### Common Issues

1. **Connection Errors**

   - Verify API keys in `.env` file
   - Check Pinecone index exists and is accessible
   - Ensure OpenAI API key has sufficient credits

2. **No Knowledge Base Content**

   - Run `initialize_knowledge_base.py` to set up initial documents
   - Check Pinecone index has vectors loaded

3. **Slow Responses**
   - Pinecone queries may have latency
   - Check OpenAI API rate limits

### Debug Information

The Streamlit app includes debug information expandable sections showing:

- Retriever setup details
- Query processing logs
- Error messages and stack traces

## Development

### Adding New Documents

Use the PDF processor to add new documents:

```python
from app.pdf_processor import PDFProcessor

processor = PDFProcessor()
success, chunk_count = processor.process_pdf_url(
    url="https://example.com/document.pdf",
    title="Document Title"
)
```

### Customizing Retrieval

Modify retrieval parameters in `app/retriever.py`:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Number of documents to retrieve
)
```

## Data Sources

The system is trained on:

- GSU 2020-21 Undergraduate Catalog
- Additional institutional documents (as configured)

All information is accurate as of the 2020-21 academic year. Users should verify current requirements with official GSU sources.

## License

This project is intended for educational purposes. GSU catalog content remains the property of Georgia State University.
