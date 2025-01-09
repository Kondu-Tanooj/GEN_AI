import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import Document class

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    documents = []
    
    # Load all CSV files from the directory
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith('.csv'):
            file_path = os.path.join(DATA_PATH, file_name)
            df = pd.read_csv(file_path)
            print(f"Columns in {file_name}: {df.columns}")  # Debugging: print columns
            
            # Check the correct column names for each file and adjust accordingly
            if file_name == 'Gita_Word_Meanings_Hindi.csv':
                if 'Sanskrit Word' in df.columns and 'Hindi Meaning' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit Word: {row['Sanskrit Word']} | Hindi Meaning: {row['Hindi Meaning']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Bhagwad_Gita_Verses_English.csv':
                if 'Sanskrit ' in df.columns and 'Swami Adidevananda' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit: {row['Sanskrit ']} | Swami Adidevananda: {row['Swami Adidevananda']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Bhagwad_Gita_Verses_English_Questions.csv':
                if 'sanskrit' in df.columns and 'question' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit: {row['sanskrit']} | Question: {row['question']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Patanjali_Yoga_Sutras_Verses_English.csv':
                if 'Sanskrit ' in df.columns and 'Word Meanings' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit: {row['Sanskrit ']} | Word Meaning: {row['Word Meanings']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Gita_Word_Meanings_English.csv':
                if 'Sanskrit Word' in df.columns and 'English Meaning' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit Word: {row['Sanskrit Word']} | English Meaning: {row['English Meaning']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Bhagwad_Gita_Verses_Concepts.csv':
                if 'Sanskrit' in df.columns and 'English' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit: {row['Sanskrit']} | Concept: {row['Concept']} | Keyword: {row['Keyword']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document

            elif file_name == 'Patanjali_Yoga_Sutras_Verses_English_Questions.csv':
                if 'sanskrit' in df.columns and 'question' in df.columns:
                    for index, row in df.iterrows():
                        combined_text = f"Sanskrit: {row['sanskrit']} | Question: {row['question']}"
                        documents.append(Document(page_content=combined_text))  # Wrap text in Document
    
    # Check if documents have been added
    if not documents:
        print("No documents were added. Please check the column names in the CSV files.")
        return

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create vector store with FAISS
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

