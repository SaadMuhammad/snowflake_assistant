import json
import os  
import pandas as pd
import numpy as np
import tiktoken
import faiss
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt 
import zipfile

#read index title
index_snwflk_title = faiss.read_index(r'./new_data/index_snwflk_title.index')

#read json title
#with open('/main/content/snwflk_title_json.json', 'r', encoding='utf-8') as file:
    #input_data_title = json.load(file)
# Read json title  
with zipfile.ZipFile(r'./new_data/snwflk_title_json.zip', 'r') as z:  
    with z.open('snwflk_title_json.json') as f:  
        input_data_title = json.load(f)

#read text json
df_text = pd.read_json(r'./new_data/snwflk_chunksplits.zip', compression='zip') 
    
#read text vector embeddings
#files = [rf'/content/snwflk_chunksplits_vectorp{i}.json' for i in range(1, 5)]  
# Read each file into a DataFrame and store all DataFrames in a list  
#dfss = [pd.read_json(file) for file in files]  
# Concatenate all DataFrames into a single DataFrame  
#df_vector = pd.concat(dfss, ignore_index=True)  
files = [r'./new_data/snwflk_chunksplits_vectorp{}.zip'.format(i) for i in range(1, 5)]  
dfss = [pd.read_json(f, compression='zip') for f in files]  
df_vector = pd.concat(dfss, ignore_index=True)

#read index
#index_snwflk = faiss.read_index('/content/index_snwflk.index')

#Read the text.json
#with open('/content/snwflk.json', 'r', encoding='utf-8') as file:
    #nput_data = json.load(file)

#call model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")

def get_info(index, title_data):
    # Handle cases where the index is out of bounds
    if index < 0 or index >= len(title_data):
        return None

    item = title_data[index]
    return {
        'URL': item['URL'],
        'Title': item['Title'],
        'Text': item['Text'],
    }

def get_temp_info(index, df):
    # Handle cases where the index is out of bounds
    if index < 0 or index >= len(df):
        return None

    # Get the item
    item = df.loc[index]

    # Check for high New_Token_Count and return empty text
    if item['New_Token_Count'] > 2000:
        return {'Title': item['Title'], 'Text': "empty"}

    # Initialize text with the item's 'Text'
    text = item['Text']

    # If the index is not the first and the previous item's 'New_Token_Count' is less than 501,
    # prepend its 'Text' to 'text'
    if index - 1 >= 0 and df.loc[index - 1, 'New_Token_Count'] < 501:
        text = df.loc[index - 1, 'Text'] + " " + text

    # If the index is not the last and the next item's 'New_Token_Count' is less than 501,
    # append its 'Text' to 'text'
    if index + 1 < len(df) and df.loc[index + 1, 'New_Token_Count'] < 501:
        text = text + " " + df.loc[index + 1, 'Text']

    return {
        'Title': item['Title'],
        'Text': text,
    }

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings_arctic(text: str):
    return model.encode(text).tolist()

#use user question to get data from faiss index
def faiss_search(query):
    embed = generate_embeddings_arctic(query)  
    ques = np.array(embed).astype('float32')  
    ques = ques.reshape(1, -1)  
    distances, indices = index_snwflk_title.search(ques, 3) 
    # Store the titles available in indices as a list
    titles = [input_data_title[i]['Title'] for i in indices[0]]
    # Get index of all matching title rows and create a list for all such index
    matching_indices = df_text[df_text['Title'].isin(titles)].index.tolist()

    # Get the embeddings from the rows that match the indices
    embeddings = df_vector.loc[matching_indices, 'Text_Embeddings'].tolist()

    # Create a subset DataFrame based on matching indices and reset the index
    subset_df = df_text.loc[matching_indices].reset_index(drop=True)

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    #print(embeddings_array.shape[1])
    index_temp = faiss.IndexFlatL2(embeddings_array.shape[1])
    #print(index_temp.is_trained)
    index_temp.add(embeddings_array)
    #print(index_temp.ntotal)

    embed = generate_embeddings_arctic(query)
    ques = np.array(embed).astype('float32')
    ques = ques.reshape(1, -1)
    distances, indices = index_temp.search(ques, 2)

    result = {}
    for i in indices[0]:
        result[i] = get_temp_info(i, subset_df)

    #context_dict = {}  
    #for i in indices[0]:    
        #info = get_info(i)   
        #context_dict[i] = info  
        #if info['New_Token_Count'] <= 1500:    
            #context_dict[i-1] = get_info(i - 1)  
            #context_dict[i+1] = get_info(i + 1)   
  
    return result
