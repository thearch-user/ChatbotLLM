from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

# Assuming model.py and model_args.py are in the parent directory or properly in PYTHONPATH
# For simplicity, let's adjust the import to work with the current structure
import sys
sys.path.append('..') # Add parent directory to path to import model

from model.model import Transformer

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",  # Default React development server port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (placeholder for now, as no pre-trained weights are provided)
try:
    model = Transformer()
    # model.load_state_dict(torch.load("path/to/your/model.pth")) # Uncomment and specify path if you have pre-trained weights
    model.eval() # Set model to evaluation mode
    print("Model initialized successfully.")
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None # Handle case where model fails to load

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(text: dict):
    if model is None:
        return {"error": "Model not loaded"}

    input_text = text.get("text", "")
    # In a real scenario, you would tokenize input_text, convert to tensors,
    # and pass through the model. For now, we return a dummy response.
    
    # Example of dummy tensor creation (replace with actual tokenization and model input)
    dummy_input = torch.randint(0, 50257, (1, 10)) # Batch size 1, sequence length 10
    
    with torch.no_grad():
        # dummy_output = model(dummy_input) # This would be the actual model call
        pass

    # For now, just echoing the input and a generic response
    print(f"Received text for prediction: {input_text}")
    return {"input": input_text, "prediction": f"Dummy prediction for: '{input_text}'"}
