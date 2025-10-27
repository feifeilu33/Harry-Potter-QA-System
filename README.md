# Harry Potter Expert: A RAG-Powered Conversational AI
This repository contains the source code and documentation for **Harry Potter Expert**—a course project developed for Large Language Model with Prompt Engineering. The project aims to build an intelligent conversational system that helps Harry Potter (HP) fans, scholars, and creators efficiently explore the HP magical universe by providing accurate, context-aware answers to their queries.


## 1. Project Overview
### 1.1 Course Background
This is a team project completed as part of my Large Language Model with Prompt Engineering course. The goal was to design and implement a domain-specific question-answering (QA) system using state-of-the-art Retrieval-Augmented Generation (RAG) technology.

### 1.2 Project Motivation
The Harry Potter series (by J.K. Rowling) is a global cultural phenomenon, with over 750 characters, 2000+ magical elements, and extensive storylines across books, movies, and games. Navigating this vast universe to find accurate information (e.g., character details, plot events) is challenging. 

**Harry Potter Expert** addresses this by:
- Providing consistent, accurate answers aligned with the original canon (books, official sources).
- Supporting fan fiction authors, film/game developers, and new readers in exploring the HP universe.
- Enabling multi-round conversations with a "naughty-wizard style" interface for an engaging user experience.


## 2. Core Features
| Feature | Description |
|---------|-------------|
| **Intelligent Question Classification** | Automatically categorizes user queries into 3 types: <br> - `Character`: Queries about character attributes (e.g., "What is Ron's wand made of?"). <br> - `Plot`: Queries about story events (e.g., "When did Harry first meet Ron?"). <br> - `Other`: Queries requiring external knowledge (e.g., "Who plays Harry in the movies?"). |
| **Multi-Source Information Retrieval** | Fetches data from 3 trusted sources based on query type: <br> - **HP Book Chunks**: Semantic search on the first HP book (*Harry Potter and the Sorcerer's Stone*). <br> - **HP Public API**: Fuzzy matching to retrieve official character metadata (name, house, wand, patronus, etc.). <br> - **Web Search**: Google Search + web scraping for external knowledge (e.g., real-world inspirations, actor info). |
| **RAG-Powered Response Generation** | Combines retrieved context with a large language model (LLM) to generate accurate, non-hallucinated answers. Maintains conversation history for context continuity. |
| **Interactive Web Interface** | Built with Streamlit, featuring a wizard-themed UI for multi-round conversations. |


## 3. Tech Stack
### 3.1 Dependencies
| Category | Tools/Libraries | Purpose |
|----------|-----------------|---------|
| **Core ML/AI** | `sentence-transformers` (all-MiniLM-L6-v2) | Text embedding for semantic search |
| | `transformers` (Llama-3.2-3B-Instruct, BAAI/BGE-Ranker-V2-Gemma) | LLM for response generation; base model for question classification |
| | `peft` (LoRA) | Parameter-efficient fine-tuning of the question classification model |
| | `torch` | GPU-accelerated model training/inference |
| **Data Processing** | `langchain.text_splitter` (RecursiveCharacterTextSplitter) | Split book text into semantic chunks |
| | `numpy` | Calculate Euclidean distance for embedding similarity |
| | `fuzzywuzzy` | Fuzzy matching for character name extraction |
| **Web & API** | `requests` | Call HP public API and fetch web content |
| | `bs4` (BeautifulSoup) | Web scraping (extract text from search results) |
| | `googlesearch` | Execute Google searches for external knowledge |
| **UI & Deployment** | `streamlit` | Build interactive web interface |
| **Utility** | `os`, `json` | File/directory operations and JSON data handling |


## 4. Methodology
The system follows a 4-step workflow to process user queries and generate answers:

### Step 1: Data Preprocessing
1. **Book Text Processing**: Load the first HP book (from `huggingface.co/datasets/WutYee/HarryPotter_books_1to7`), clean text (lowercasing, whitespace removal), split into 1000-token chunks (100-token overlap) using `RecursiveCharacterTextSplitter`.
2. **Embedding Generation**: Encode book chunks into vectors with `all-MiniLM-L6-v2` and save to `chunks.json` (text) and `.npy` (embeddings) for fast retrieval.
3. **Character Data**: Fetch official character metadata from the [HP Public API](https://hp-api.onrender.com/api/characters).

### Step 2: Question Classification
- **Model**: Fine-tune `BAAI/BGE-Ranker-V2-Gemma` with **LoRA** (Low-Rank Adaptation) on a manually labeled HP-specific dataset (to balance `Character`/`Plot`/`Other` categories).
- **Output**: A category ID (1=Character, 2=Plot, 3=Other) to route the query to the correct retrieval module.

### Step 3: Multi-Source Retrieval
| Query Type | Retrieval Logic |
|------------|-----------------|
| Character (ID=1) | Use `fuzzywuzzy` to extract character names from the query → Fetch detailed attributes from the HP API. |
| Plot (ID=2) | Encode the query into a vector → Calculate Euclidean distance with precomputed book chunk embeddings → Retrieve top-5 most similar chunks. |
| Other (ID=3) | Run Google Search → Scrape text from top-3 URLs with `BeautifulSoup` → Extract relevant content. |

### Step 4: RAG Response Generation
1. **Prompt Construction**: Inject retrieved context, user query, and conversation history into a structured prompt (to ground the LLM and avoid hallucinations).
2. **LLM Inference**: Use `Llama-3.2-3B-Instruct` (via `transformers.pipeline`) to generate natural, context-aware answers.
3. **UI Output**: Display the answer in the Streamlit interface, along with conversation history.


## 5. Quick Start
### 5.1 Environment Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/harry-potter-expert.git
   cd harry-potter-expert
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (See `requirements.txt` for full list of dependencies, including `streamlit`, `sentence-transformers`, `peft`, etc.)

### 5.2 Data Preparation
1. **Book Data**: The first HP book text is automatically downloaded from Hugging Face Datasets during initialization.
2. **Embeddings**: Run `src/book_processor.py` to generate text chunks and embeddings:
   ```bash
   python src/book_processor.py
   ```
3. **API Access**: No API keys required (HP Public API is free; Google Search requires a valid API key for production use).

### 5.3 Run the Application
Start the Streamlit web interface:
```bash
streamlit run app.py
```
Access the UI at `http://localhost:8501` in your browser.


## 6. Project Structure
```
harry-potter-expert/
├── data/                  # Processed data (book chunks, embeddings)
│   ├── chunks.json        # Book text chunks
│   └── embeddings.npy     # Embeddings of book chunks
├── src/                   # Core modules
│   ├── book_processor.py  # Book text preprocessing + embedding generation
│   ├── text_classifier.py # Question classification (LoRA-fine-tuned model)
│   ├── query_processor.py # Plot-related semantic search
│   ├── hp_character_info.py # Character data retrieval from HP API
│   └── web_searcher.py    # Web search + scraping
├── app.py                 # Streamlit UI entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation (this file)
```


## 7. Example Usage
| User Query | System Behavior | Output |
|------------|------------------|--------|
| "When did Harry first meet Ron?" | 1. Classified as `Plot` (ID=2). <br> 2. Retrieves top-5 book chunks about their first meeting. <br> 3. Generates answer via Llama-3.2-3B-Instruct. | "Harry first met Ron Weasley on the Hogwarts Express on the way to their first year at Hogwarts." |
| "What is Ron Weasley's birthday?" | 1. Classified as `Character` (ID=1). <br> 2. Fuzzy-matches "Ron Weasley" → Fetches API data. <br> 3. Formats metadata into a readable answer. | "Ron Weasley's date of birth is 01-03-1980. He was born into a pure-blood wizarding family and sorted into Gryffindor House." |
| "Who plays Harry Potter in the movies?" | 1. Classified as `Other` (ID=3). <br> 2. Runs Google Search → Scrapes Wikipedia. <br> 3. Extracts actor info. | "Daniel Radcliffe portrays Harry Potter in the HP film series." |


## 8. Limitations & Future Work
### 8.1 Current Limitations
- **Limited Book Coverage**: Only supports the first HP book (*Harry Potter and the Sorcerer's Stone*).
- **English-Only**: No multilingual support (datasets and models are English-focused).
- **Simplified Classification**: The 3-category system struggles with nuanced queries (e.g., "What magical object did Harry use to defeat Voldemort?").
- **No Causal Analysis**: Cannot answer "what-if" questions (e.g., "How would Harry’s life change without Hagrid?").

### 8.2 Future Improvements
1. **Expand Book Coverage**: Include all 7 HP books to support broader plot queries.
2. **Multilingual Support**: Integrate cross-language models (e.g., mBERT) and translate datasets.
3. **Enhance Classification**: Add subcategories (e.g., "Magical Objects", "Character Motivations") validated via user testing.
4. **Causal Reasoning**: Implement event-based knowledge graphs to answer "what-if" questions.


## 9. Acknowledgements
- J.K. Rowling for creating the Harry Potter universe.
- The HP Public API (https://hp-api.onrender.com) for official character data.
- Hugging Face for open-source datasets (HP books) and pre-trained models.
