
# import os

import torch
print(torch.cuda.is_available())


# Load model directly

from transformers import AutoTokenizer, AutoModelForCausalLM

parh = "/home/credentek/Rajesh/LLMmodel/sports_model"
print("load model ...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")  
# tokenizer = AutoTokenizer.from_pretrained(parh)
# model = AutoModelForCausalLM.from_pretrained(parh)  

# parh = "/home/credentek/Rajesh/LLMmodel/model"

# models = model.save_pretrained(parh)
# tokenizers = tokenizer.save_pretrained(parh)

print("loaded ...")

from fastapi import FastAPI

app = FastAPI()


@app.post("/generate")
def generate_text(question: str):
    # sports_keywords = [
    #     "cricket", "football", "soccer", "hockey", "tennis",
    #     "badminton", "basketball", "kabaddi", "olympics",
    #     "match", "tournament", "player", "score", "goal"
    # ]

    # # Check if question is sports-related
    # if not any(word in question.lower() for word in sports_keywords):
    #     return {"system": "Questions only sports-related","question" : question}
    
    prompt = f"You are a sports bot.\nUser: {question}\n"

    inputs = tokenizer(question, return_tensors="pt")
    print("input",inputs)

    output = model.generate(**inputs,max_length=200,temperature=0.7,do_sample=True,no_repeat_ngram_size=2,top_p=0.9)
    
    print(output)
    
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result
 

import uvicorn
uvicorn.run(app,port = 8000)

