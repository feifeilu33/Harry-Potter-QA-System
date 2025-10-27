# pip install numpy sentence-transformers langchain transformers peft torch fuzzywuzzy python-Levenshtein googlesearch-python beautifulsoup4 requests streamlit

# ===========================
# Step 1: Import necessary libraries
# ===========================

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import torch
from fuzzywuzzy import process
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import streamlit as st
from transformers import pipeline

# ===========================
# Step 2: Define file paths
# ===========================

book_path = r"D:\6620_combined\HP1.txt" 
processed_text_path = r"D:\6620_combined\processed_text.txt" #"your_path\processed_text.txt"
output_dir = r"D:\6620_combined"  

# ===========================
# Step 3: Define the BookProcessor class
# This class handles loading, preprocessing, chunking, and embedding of the book text.
# Here we only access Harry Potter and the Sorcerer's Stone (HP1).
# As all the Harry Potter books text file are different, making it difficult to clean and process them all at once.
# ===========================

class BookProcessor:
    def __init__(self, book_path, processed_text_path, output_dir):
        """
        Initialize the BookProcessor with paths and model.
        """
        self.book_path = book_path
        self.processed_text_path = processed_text_path
        self.output_dir = output_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_text(self, text):
        """
        Preprocess the text by performing basic cleaning operations.
        """
        print("Preprocessing text...")
        text = text.strip() 
        text = " ".join(text.split())  
        text = text.lower() 
        return text

    def load_and_preprocess_book(self):
        """
        Load book content, preprocess it, and save the processed text.
        """
        print("Loading and preprocessing the book...")
        with open(self.book_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Call the preprocess_text method
        processed_text = self.preprocess_text(text)
        
        # Save the processed text to a .txt file
        with open(self.processed_text_path, 'w', encoding='utf-8') as file:
            file.write(processed_text)
        
        print(f"Processed text saved to: {self.processed_text_path}")
        return processed_text

    def chunk_and_embed(self, text):
        """
        Use RecursiveCharacterTextSplitter to split the text into chunks and generate embeddings.
        """
        print("Chunking and generating embeddings...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Maximum number of characters per chunk
            chunk_overlap=100,     # Number of overlapping characters between chunks
            separators=["\n\n", " "],  # List of separators in order of priority
        )
        
        chunks = text_splitter.split_text(text)  # Split the text
        embeddings = self.model.encode(chunks)  # Generate embeddings
        
        return {
            "chunks": chunks,
            "embeddings": embeddings
        }

    def save_intermediate_results(self, data):
        """
        Save chunks and embeddings to the specified directory.
        """
        print("Saving intermediate results...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'chunks.json'), 'w', encoding='utf-8') as file:
            json.dump(data['chunks'], file, ensure_ascii=False)
        
        np.save(os.path.join(self.output_dir, 'embeddings.npy'), data['embeddings'])
        print(f"Intermediate results saved to: {self.output_dir}")

    def process(self):
        """
        Main process to load, preprocess, chunk, embed, and save results.
        """
        # Step 1: Load and preprocess the book
        processed_text = self.load_and_preprocess_book()
        
        # Step 2: Chunk and generate embeddings
        result = self.chunk_and_embed(processed_text)
        
        # Step 3: Save intermediate results
        self.save_intermediate_results(result)
        
        print("Processing completed successfully!")

# ===========================
# Step 4: Define the TextClassifier class
# This class handles loading the fine-tuned text classification model and classifying input text.
# We used a set of questions and coresponding labels to fine-tune the model.
# ===========================

class TextClassifier:
    def __init__(self, text_classification_model, finetune_path, labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            text_classification_model,
            num_labels=len(labels),
            torch_dtype=torch.float16 if self.device.type == "cuda" else None,
        ).to(self.device)
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, finetune_path).to(self.device)
        
        # Merge LoRA weights to optimize inference speed
        self.model = self.model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(text_classification_model)       
        self.labels = labels

    def classify_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        max_index = scores.argmax().item()
        return self.labels[max_index], scores.tolist()

    def classify_question(self, text):
        category, scores = self.classify_text(text)
        if category == "character":
            return 1
        elif category == "plot":
            return 2
        else:
            return 3

# ===========================
# Step 5: Define the QueryProcessor class
# If the query is related to the plot, it will call this class.
# This class handles loading the intermediate results (chunks and embeddings),
# generating the query embedding, finding the most similar chunks based on Euclidean distance,
# and return the top-k most relevant chunks.
# ===========================

class QueryProcessor:
    def __init__(self, output_dir, query_embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the QueryProcessor with paths and model.
        """
        self.output_dir = output_dir
        self.model = SentenceTransformer(query_embedding_model)

    def load_intermediate_results(self):
        """
        Load chunks and embeddings from the specified directory.
        """
        print("Loading intermediate results...")
        with open(os.path.join(self.output_dir, 'chunks.json'), 'r', encoding='utf-8') as file:
            chunks = json.load(file)
        
        embeddings = np.load(os.path.join(self.output_dir, 'embeddings.npy'))
        return chunks, embeddings

    def generate_query_embedding(self, query):
        """
        Generate an embedding for the given query using the provided model.
        """
        print("Generating query embedding...")
        query_embedding = self.model.encode(query)
        return query_embedding

    def find_most_similar_chunks(self, query_embedding, embeddings, chunks, top_k=5):
        """
        Find the most similar chunks to the query based on Euclidean distance.
        """
        print("Finding most similar chunks...")
        # Calculate Euclidean distances between query and all embeddings
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        
        # Get indices of top-k most similar chunks (smallest distances)
        top_indices = np.argsort(distances)[:top_k]
        
        # Retrieve the corresponding chunks and their distance scores
        top_chunks = [(chunks[i], distances[i]) for i in top_indices]
        return top_chunks

    def process_query(self, query, top_k=5):
        """
        Process a query to find the most relevant chunks.
        """
        # Step 1: Load intermediate results
        chunks, embeddings = self.load_intermediate_results()
        
        # Step 2: Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Step 3: Find most similar chunks
        top_chunks = self.find_most_similar_chunks(query_embedding, embeddings, chunks, top_k)
        
        return top_chunks

    def get_topk_text(self, query, top_k=5):
        """
        Get the top-k most relevant chunks as a single concatenated string.
        """
        # Process the query to get the top-k chunks
        top_chunks = self.process_query(query, top_k)
        
        # Extract only the text of the chunks
        top_texts = [chunk for chunk, _ in top_chunks]
        
        # Concatenate the texts into a single string
        concatenated_text = " ".join(top_texts)
        
        return concatenated_text

# ===========================
# Step 6: Define the HPCharacterInfo class
# If the query is related to a character, it will call this class.
# This class handles fetching character data from the API.
# It uses fuzzy matching to identify character names from the question.
# If the character name is found, it retrieves detailed information about that character.
# ===========================

class HPCharacterInfo:
    def __init__(self, api_url="https://hp-api.onrender.com/api/characters"):
        self.api_url = api_url

    def fetch_characters_data(self):
        """Fetch character data from the API."""
        response = requests.get(self.api_url)
        if response.status_code == 200:
            return response.json()  # Return JSON data
        else:
            raise Exception("Failed to fetch character data. Please check if the API is available.")

    def extract_character_name(self, question, characters):
        """Extract character name from the question using fuzzy matching."""
        # Extract all character names
        character_names = [character["name"] for character in characters if character.get("name")]
        
        # Use fuzzywuzzy's process.extractOne to find the best match
        match, score = process.extractOne(question, character_names)
        if score > 70:  # Set a threshold to avoid mismatches
            return match
        return None

    def get_character_info(self, character_name, characters):
        """Get detailed information about a character."""
        for character in characters:
            if character.get("name") == character_name:
                return character
        return None

    def answer_question(self, question):
        """Answer the question by fetching and processing character data."""
        try:
            # Call API to fetch character data
            characters = self.fetch_characters_data()
            
            # Extract character name
            character_name = self.extract_character_name(question, characters)
            
            if character_name:
                # Get detailed character information
                info = self.get_character_info(character_name, characters)
                
                if info:
                    # Format the output
                    response = f"Information about {info['name']}:\n"
                    for key, value in info.items():
                        if value is not None:  # Ignore fields with null values
                            response += f"- {key.capitalize()}: {value}\n"
                    return response
                else:
                    return f"Details about {character_name} were not found."
            else:
                return "Sorry, I couldn't identify the character name in the question."
        except Exception as e:
            return f"An error occurred: {str(e)}"

# ===========================
# Step 7: Define the WebSearcher class
# If the query is not related to character or plot, it will call this class.
# This class handles performing a web search using the Google search API,
# fetching the webpage content, and cleaning it for LLM input.
# ===========================

class WebSearcher:
    def __init__(self, query: str, num_results: int = 3):
        """
        Initialize the WebSearcher with a query and number of results to fetch.
        
        Args:
            query (str): The search query.
            num_results (int): Number of search results to process.
        """
        self.query = query
        self.num_results = num_results
        self.results = []

    def perform_search(self):
        """
        Perform a web search and store the results as a list of dictionaries.
        Each dictionary contains the URL and cleaned content of a webpage.
        """
        try:
            # Perform the search and iterate over the top results
            for url in search(self.query, num_results=self.num_results):
                try:
                    # Fetch the webpage content
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract text content and clean it
                    body = soup.get_text(separator="\n")  # Use newline as separator for readability
                    body = " ".join(body.split())  # Remove extra whitespace
                    
                    if body.strip():  # Only add if the content is not empty
                        self.results.append({
                            "url": url,
                            "content": body[:1000]  # Limit content to the first 1000 characters
                        })
                except Exception as e:
                    print(f"Error processing webpage {url}: {e}")
        except Exception as e:
            print(f"Error during web search: {e}")

    def get_llm_input(self) -> str:
        """
        Combine all content from the search results into a single string for LLM input.
        
        Returns:
            str: A single string containing all content from the search results.
        """
        return "\n".join(result["content"] for result in self.results)

# ===========================
# Step 8: Define the main query processing function
# This function processes the input query, classifies it, and returns the appropriate information.
# It uses the TextClassifier to determine the type of query (character, plot, or other),
# and then calls the appropriate class (HPCharacterInfo or QueryProcessor or WebSearcher)
# to handle the query accordingly.
# ===========================

def process_query_and_get_info(query):
    """
    Process the input query, classify it, and return the appropriate information.
    
    Args:
        query (str): The input query.
    
    Returns:
        str: The response based on the query type.
    """
    # Step 1: Initialize the TextClassifier with required parameters
    text_classifier = TextClassifier(
        text_classification_model= "BAAI/bge-reranker-v2-gemma",  # Replace with your model
        finetune_path=r"D:\6620_combined\checkpoint-378",   # Replace with your fine-tuned model path
        labels=["character", "plot", "other"]          # Define the labels
    )
    
    # Step 2: Classify the query
    query_type_id = text_classifier.classify_question(query)
    
    # Step 3: Handle the query based on its type
    if query_type_id == 1:  # Character-related question
        hp_character_info = HPCharacterInfo()
        response = hp_character_info.answer_question(query)
    elif query_type_id == 2:  # Plot-related question
        query_processor = QueryProcessor(output_dir=r"D:\6620_hp\intermediate_HP1")  # Replace with your output directory
        response = query_processor.get_topk_text(query, top_k=5)
    else:  # Other types of questions
        web_searcher = WebSearcher(query=query, num_results=3)
        web_searcher.perform_search()
        response = web_searcher.get_llm_input()
    
    return query_type_id, response

# ===========================
# Step 9: Streamlit app setup
# This part initializes the Streamlit app and processes the book once to prepare it for queries.
# This is only run once to preprocess the book text and generate embeddings.
# It ensures that the book is processed before any queries are made.
# ===========================

@st.cache_data
def process_book_once():
    book_processor = BookProcessor(book_path, processed_text_path, output_dir)
    book_processor.process()
process_book_once()

# ===========================
# Step 10: Streamlit UI logic
# This is the main part of the Streamlit app where user input is handled,
# messages are displayed, and responses are generated.
# It maintains the last few messages in the session state to keep the conversation context alive.
# and updates the UI with user queries and assistant responses.
# ===========================

st.title("Harry Potter Wizard Land")
if "messages" not in st.session_state:
    st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a wizard of Harry Potter and a helpful but naughty assistant.",
            },
    ]

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(f"You: {message['content']}")
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(f"Assistant: {message['content']}")

user_input = st.chat_input("Ah, curious soul, what secrets do ye wish to uncover?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(f"You: {user_input}")

    query_type_id, info = process_query_and_get_info(user_input)
    
    category_map = {1: "Character-related", 2: "Plot-related", 3: "Other"}
    category_name = category_map.get(query_type_id, "Unknown")
    st.write(f"**Query classified as:** {category_name} (Category ID: {query_type_id})")

    if info:
        info = str(info)
        st.session_state.messages.append({"role": "system", "content": info})

    recent_messages = st.session_state.messages[-5:]
    outputs = pipe(
        recent_messages,
        max_new_tokens=100,
    )
    print(outputs)

    anwser = outputs[0]["generated_text"][-1]["content"]
    print(anwser)

    st.session_state.messages.pop() # remove the last system message info
    st.session_state.messages.append({"role": "assistant", "content": anwser})
    st.chat_message("assistant").write(f"Assistant: {anwser}")