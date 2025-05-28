# AI Advisor Web Scraper

This application scrapes web pages, chunks the content, generates embeddings using OpenAI, and stores them in Pinecone for semantic search.

## Setup Instructions

### 1. Install Pinecone Package

First, you need to install the correct Pinecone package. Since we couldn't install it during the initial setup due to compilation issues, try:

```bash
# Activate your virtual environment first
.\venv\bin\Activate.ps1

# Install pinecone (the correct package name)
pip install pinecone-client
```

### 2. Set Up API Keys

1. **Copy the environment template:**

   ```bash
   copy app\env_template.txt .env
   ```

2. **Get your API keys:**

   **OpenAI API Key:**

   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create an account if you don't have one
   - Generate a new API key
   - Copy it to your `.env` file

   **Pinecone API Key:**

   - Go to [Pinecone Console](https://app.pinecone.io/)
   - Create a free account
   - Create a new project
   - Copy your API key and environment from the dashboard
   - Add them to your `.env` file

3. **Update your `.env` file:**
   ```env
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   PINECONE_API_KEY=your-actual-pinecone-key-here
   PINECONE_ENVIRONMENT=your-pinecone-environment
   PINECONE_INDEX_NAME=ai-advisor-index
   ```

### 3. Add Your URLs

Edit `app/scraper.py` and add your URLs to the `urls` list in the `main()` function:

```python
def main():
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://your-website.com/important-page",
        # Add more URLs here
    ]
```

### 4. Run the Scraper

```bash
cd app
python scraper.py
```

This will:

- Scrape all the URLs you specified
- Split content into chunks using RecursiveCharacterTextSplitter
- Generate OpenAI embeddings for each chunk
- Save everything to `scraped_data.json`

### 5. Initialize Pinecone Database

```bash
python pinecone_setup.py
```

This will:

- Create a Pinecone index (if it doesn't exist)
- Upload all your vectorized data to Pinecone
- Display statistics about your index

## File Structure

```
app/
├── scraper.py          # Main scraping and embedding script
├── pinecone_setup.py   # Pinecone database initialization
├── env_template.txt    # Environment variables template
├── README.md           # This file
└── scraped_data.json   # Generated data (after running scraper)
```

## Usage Examples

### Basic Scraping

```python
from scraper import WebScraper

# Initialize scraper
scraper = WebScraper(chunk_size=1000, chunk_overlap=200)

# Process URLs
urls = ["https://example.com"]
chunks = scraper.process_urls(urls)

# Save data
scraper.save_to_json("my_data.json")
```

### Pinecone Operations

```python
from pinecone_setup import PineconeManager

# Initialize manager
manager = PineconeManager()

# Create index
manager.create_index()

# Upload vectors
vectors = load_scraped_data("scraped_data.json")
manager.upload_vectors(vectors)

# Query similar content
query_vector = [0.1, 0.2, ...]  # Your query embedding
results = manager.query_index(query_vector, top_k=5)
```

## Features

- **Smart Content Extraction:** Automatically finds main content areas
- **Flexible Chunking:** Configurable chunk size and overlap
- **Rate Limiting:** Built-in delays to respect API limits
- **Error Handling:** Robust error handling for failed requests
- **Batch Processing:** Efficient batch uploads to Pinecone
- **Metadata Storage:** Rich metadata for each chunk

## Configuration Options

### Scraper Settings

- `chunk_size`: Maximum characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### Pinecone Settings

- `dimension`: Vector dimension (default: 1536 for OpenAI ada-002)
- `metric`: Distance metric (default: "cosine")
- `batch_size`: Upload batch size (default: 100)

## Troubleshooting

### Common Issues

1. **"No module named 'pinecone'"**

   ```bash
   pip install pinecone-client
   ```

2. **"OpenAI API key not found"**

   - Check your `.env` file
   - Make sure the file is in the correct location
   - Verify the key format starts with "sk-"

3. **"SSL Certificate errors"**

   - Try updating pip: `python -m pip install --upgrade pip`
   - Or add `--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org` to pip commands

4. **Rate limit errors**
   - The scraper includes delays, but you might need to increase them
   - Check your OpenAI usage quotas

### Performance Tips

- Start with a small set of URLs to test
- Use smaller chunk sizes for better precision
- Monitor your OpenAI API usage and costs
- Consider using Pinecone's free tier limits

## Next Steps

After setting up the scraper and database:

1. **Build a Query Interface:** Create a script to search your knowledge base
2. **Add More Content Types:** Extend to handle PDFs, documents, etc.
3. **Implement Caching:** Cache embeddings to avoid regenerating
4. **Add Update Functionality:** Handle content updates and deduplication
5. **Create a Web Interface:** Build a Streamlit app for easy searching

## Support

If you encounter issues:

1. Check the error messages carefully
2. Verify all API keys are correct
3. Make sure your virtual environment is activated
4. Check API quotas and rate limits
