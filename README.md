# Insurellm RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application built with LangChain, Chroma, and Gradio. This application allows users to ask questions about Insurellm company information, employees, products, and contracts using semantic search.

## 🚀 Features

- **Automatic Vector Store Building**: Automatically builds the vector database on first launch
- **Semantic Search**: Uses Chroma vector database for intelligent document retrieval
- **Conversational Memory**: Maintains conversation context across queries
- **Beautiful UI**: Modern Gradio interface with examples and controls
- **Production Ready**: Includes error handling and logging

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Conda (recommended) or pip

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:deXterbed/Insurellm-RAG-Assistant.git
   cd Insurellm-RAG-Assistant
   ```

2. **Create a conda environment** (recommended):
   ```bash
   conda create -n llms python=3.11
   conda activate llms
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```
   Or manually create `.env` and add:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key
   ```

## 🎯 Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Access the interface**:
   - The app will automatically open in your browser
   - Or visit `http://localhost:7860`

3. **First Launch**:
   - On first launch, the app will automatically build the vector database from the `knowledge-base/` directory
   - This may take a few minutes depending on the number of documents
   - Subsequent launches will use the existing vector database

4. **Ask Questions**:
   - Type your question in the chat interface
   - Examples:
     - "Who is Avery Lancaster?"
     - "What is Carllm?"
     - "Tell me about Insurellm"
     - "What contracts does Insurellm have?"

## 📁 Project Structure

```
week5/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # This file
├── .env                  # Environment variables (not in git)
├── knowledge-base/       # Source documents
│   ├── company/          # Company information
│   ├── employees/        # Employee profiles
│   ├── products/         # Product descriptions
│   └── contracts/        # Contract documents
└── vector_db/           # Vector database (auto-generated)
```

## 🏗️ Architecture

### Components

1. **Document Loading**: Loads markdown files from `knowledge-base/`
2. **Text Splitting**: Splits documents into chunks (1000 chars, 200 overlap)
3. **Embedding**: Uses OpenAI embeddings to create vector representations
4. **Vector Store**: Stores embeddings in Chroma database
5. **Retrieval**: Semantic search finds relevant document chunks
6. **Generation**: GPT-4o-mini generates answers based on retrieved context
7. **Memory**: Maintains conversation history for context

### Flow

```
User Question
    ↓
Embedding (Query Vector)
    ↓
Vector Similarity Search (Chroma)
    ↓
Retrieve Top-K Relevant Chunks
    ↓
Combine with Chat History
    ↓
LLM (GPT-4o-mini) Generation
    ↓
Response + Update Memory
```

## 🔧 Configuration

### Model Settings

Edit `app.py` to change:
- `MODEL`: Change the OpenAI model (default: "gpt-4o-mini")
- `temperature`: Adjust creativity (default: 0.7)
- `search_kwargs`: Change number of retrieved chunks (default: k=4)

### Vector Store Settings

Edit the `RecursiveCharacterTextSplitter` parameters:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

## 📝 Example Queries

- **Employee Information**: "Who is Avery Lancaster?", "Tell me about Alex Chen"
- **Product Information**: "What is Carllm?", "Describe Rellm"
- **Company Information**: "What is Insurellm?", "Tell me about the company"
- **Contracts**: "What contracts does Insurellm have?", "Who is TechDrive Insurance?"

## 🐛 Troubleshooting

### Vector Store Not Building
- Ensure `knowledge-base/` directory exists with markdown files
- Check that you have write permissions in the project directory
- Review logs for specific error messages

### API Key Issues
- Verify `.env` file exists and contains `OPENAI_API_KEY`
- Check that your API key is valid and has credits
- Ensure no extra spaces or quotes in the `.env` file

### Memory Issues
- Use "Reset Memory" button to clear conversation history
- Restart the app if memory becomes too large

## 📚 Technologies Used

- **LangChain**: RAG pipeline and chain orchestration
- **Chroma**: Vector database for embeddings
- **OpenAI**: Embeddings and LLM (GPT-4o-mini)
- **Gradio**: Web interface
- **Python**: Core language

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests!

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Built as part of LLM tutorials
- Uses LangChain's ConversationalRetrievalChain
- Inspired by RAG best practices

---

**Note**: Make sure to keep your `.env` file private and never commit it to version control!
