# AI-Enhanced Research Assistant

A powerful document research tool that combines vector search with AI-powered analysis using the Anura API.

## Features

### 📄 PDF Document Processing
- Upload multiple PDF files simultaneously
- Automatic text extraction and chunking
- Secure local storage using ChromaDB vector database

### 🔍 Smart Search Capabilities
1. **Basic Document Search**: Vector similarity search through your documents
2. **AI-Enhanced Search**: Basic search with AI-powered summaries (optional)
3. **Advanced RAG Search**: Comprehensive AI analysis with insights and recommendations

### 🤖 Anura API Integration
This application integrates with the Anura API to provide enhanced search results using Large Language Models:

- **AI-powered document analysis and summarization**
- **Intelligent answers to research questions**
- **Cross-document insights and connections**
- **Research gap identification and recommendations**

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Run the application:
```bash
poetry run python app.py
```

3. Open your browser to the provided URL (usually http://127.0.0.1:7860)

## Using Anura API

1. Get your API key from: https://anura-testnet.lilypad.tech/
2. Enter your API key in the search forms to enable AI features
3. Enjoy intelligent, context-aware search results!

## How It Works

1. **Document Upload**: PDFs are processed and split into chunks
2. **Vector Storage**: Text chunks are embedded and stored in ChromaDB
3. **Search**: Query similarity matching finds relevant document sections
4. **AI Enhancement**: Anura API analyzes results to provide comprehensive insights

## Search Types

- **Document Search**: Shows relevant excerpts with similarity scores and optional AI summary
- **Advanced RAG Search**: Comprehensive AI analysis including:
  - Direct answers to your questions
  - Key findings and supporting evidence
  - Cross-document insights and analysis
  - Research gaps and recommendations
  - Suggested next steps

## Technology Stack

- **Gradio**: Web interface
- **ChromaDB**: Vector database for document storage
- **PyPDF2**: PDF text extraction
- **LangChain**: Text splitting and processing
- **OpenAI API Client**: For Anura API communication
- **Anura API**: LLM-powered document analysis

## Security

- All documents are processed and stored locally
- API keys are handled securely (not stored)
- No document content is permanently stored on external servers

## Example Usage

1. **Upload Documents**: Use the "PDF Upload" tab to upload your research papers
2. **Basic Search**: Use "Document Search" tab for quick searches with optional AI enhancement
3. **Deep Analysis**: Use "Advanced RAG Search" tab for comprehensive AI-powered research analysis

The AI integration allows you to ask complex questions like:
- "What are the main conclusions across all these papers?"
- "Compare the methodologies used in different studies"
- "What gaps exist in the current research?"
- "Summarize the key findings related to [specific topic]"
