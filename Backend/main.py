from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the tokenizer and ONNX model
try:
    model_name = "prajjwal1/bert-tiny"  # Replace with the correct model name if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print("Error loading tokenizer:", e)
    raise HTTPException(status_code=500, detail="Error loading tokenizer")

try:
    onnx_model_path = "bert_spam_detection.onnx"  # Ensure this path is correct
    session = ort.InferenceSession(onnx_model_path)
    print("ONNX model loaded successfully.")
except Exception as e:
    print("Error loading ONNX model:", e)
    raise HTTPException(status_code=500, detail="Error loading ONNX model")

# Define data model for request
class EmailRequest(BaseModel):
    content: str

# Define the function to perform inference
def classify_email_onnx(text):
    try:
        # Tokenize input text
        inputs = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        print("Tokenized input:", inputs)

        # Convert input_ids and attention_mask to int64
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        print("ONNX model inputs:", onnx_inputs)

        # Run inference
        logits = session.run(None, onnx_inputs)[0]

        # Get prediction
        predicted_label = np.argmax(logits, axis=1)[0]
        return "Spam" if predicted_label == 1 else "Ham"
    except Exception as e:
        print("Error during model inference:", e)
        raise HTTPException(status_code=500, detail="Error processing the model")

# Endpoint to classify email
@app.post("/classify-email/")
async def classify_email(request: EmailRequest):
    try:
        # Perform inference
        classification = classify_email_onnx(request.content)
        return {"classification": classification}
    except Exception as e:
        print("Error in classify_email endpoint:", e)
        raise HTTPException(status_code=500, detail="Error processing the model") from e
