import json
import os  
import numpy as np
import tiktoken
import faiss
from sentence_transformers import SentenceTransformer

#read index
index_snwflk = faiss.read_index('/content/index_snwflk.index')

#Read the text.json
with open('/content/snwflk.json', 'r', encoding='utf-8') as file:
    input_data = json.load(file)

#call model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")

def get_info(index):  
    # Handle cases where the index is out of bounds  
    if index < 0 or index >= len(input_data):  
        return None  
  
    item = input_data[index]  
    return {  
        'URL': item['URL'],  
        'Title': item['Title'],  
        'Text': item['Text'],  
        'Metadata': item['Metadata']  
    }  

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings_arctic(text: str):
    return model.encode(text).tolist()

#use user question to get data from faiss index
def faiss_search(query):
    embed = generate_embeddings_arctic(query)  
    ques = np.array(embed).astype('float32')  
    ques = ques.reshape(1, -1)  
    distances, indices = index_snwflk.search(ques, 2)  

    context_dict = {}  
    for i in indices[0]:    
        info = get_info(i)   
        context_dict[i] = info  
        if info['New_Token_Count'] <= 1500:    
            context_dict[i-1] = get_info(i - 1)  
            context_dict[i+1] = get_info(i + 1)   
  
    return context_dict
