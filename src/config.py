import os
from dotenv import load_dotenv
import torch
load_dotenv()

class Config:
    # API Settings
    OLLAMA_HOST = "https://ollama.com"
    OLLAMA_API_KEY = os.getenv("OLLAMA_API")
    OLLAMA_MODEL = "qwen3-vl:235b"
    
    # Device Settings
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Processing Thresholds
    CONF_THRESHOLD = 0.70
    PADDING_Y = 10
    
    # Output Resolution
    PDF_DPI = 300