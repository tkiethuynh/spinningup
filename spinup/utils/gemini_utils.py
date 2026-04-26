import os
from google import genai
from typing import Optional

def get_gemini_client(api_key: Optional[str] = None):
    """
    Returns a Google GenAI client initialized with the provided API key
    or the GEMINI_API_KEY environment variable.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Warning: GEMINI_API_KEY not found. Gemini features will be disabled.")
        return None
    return genai.Client(api_key=key)

def analyze_training_progress(client, epoch_data: dict):
    """
    Uses Gemini to provide a natural language summary of RL training progress.
    """
    if not client:
        return "Gemini client not initialized."
    
    prompt = f"Analyze this RL training epoch data and provide a concise summary: {epoch_data}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text
