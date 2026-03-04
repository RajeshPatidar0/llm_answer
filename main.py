
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
# -----------------------------

# Device Setup

# -----------------------------

print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------

# Load Model

# -----------------------------

print("Loading model...")

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
model_name,
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

print("Model Loaded!")

# -----------------------------

# FastAPI Setup

# -----------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -----------------------------

# Request Schema

# -----------------------------

class Question(BaseModel):
    question: str

# -----------------------------

# Routes

# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
def generate_text(data: Question):
    try:
        prompt = f"You are a helpful sports assistant.\nUser: {data.question}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2
        )

        result = tokenizer.decode(output[0], skip_special_tokens=True)

        return JSONResponse(content={"response": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
