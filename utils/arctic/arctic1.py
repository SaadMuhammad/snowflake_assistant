import json
import os  
#add article model and related models
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.linear.config import QuantizationConfig

tokenizer = AutoTokenizer.from_pretrained(
    "Snowflake/snowflake-arctic-instruct",
    trust_remote_code=True
)
quant_config = QuantizationConfig(q_bits=8)

model = AutoModelForCausalLM.from_pretrained(
    "Snowflake/snowflake-arctic-instruct",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    ds_quantization_config=quant_config,
    max_memory={i: "150GiB" for i in range(8)},
    torch_dtype=torch.bfloat16)

#messages = [{"role": "user", "content": content}]
#input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

#outputs = model.generate(input_ids=input_ids, max_new_tokens=256)
#print(tokenizer.decode(outputs[0]))
#https://huggingface.co/Snowflake/snowflake-arctic-instruct


def generate_ai_response(data, question: str):
    client = Snowflake_arctic(
        )
    response = client.generate(
    model="gpt-4", 
    messages=[
        {"role": "system", "content": "You are a snowflake assistant support specialist. I will provide you with context and question, \
         The question contain a semantic answer searched from documents and related matches.\
         sometime context don't have a semantic answer only related matches. So your role is to analyze the context and user question and determine \
         if there is any useful information in context that can be used answer user question.\
         if yes, answer user question cleary and as helpfully as possible and extend asnwer where needed but don't add anythong from outside the given context data. \
         if the question and context don't make sense simply answer this is out of the documentation scope, please reach out to \
         SnowFlake team. Lastly, when you find a relevant answer in context then always provide the website link from relevant match for user \
         to further read and verify. If there are multiple correct answers in related matches,\
         then combine the answer and share all relevant links. Ypu will never write 'based on given context' or 'based on a given related match number' or \
         'I don't find similar semantic answer' in response to user question\
          if you don't find answer and simply refer him to contact snowflake team"},
        {"role": "user", "content": f"Context: here's the data for context {data}, if a title matches do provide the relevant web link to user even if content doesn't have complete answer\
         Now this is the user's question: Question: {question}"}
    ], )
    return response