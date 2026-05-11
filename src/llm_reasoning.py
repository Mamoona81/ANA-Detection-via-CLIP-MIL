import os
import google.generativeai as genai
from PIL import Image

def init_gemini():
    """
    Initializes Gemini API using the GOOGLE_API_KEY environment variable.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set. Gemini API will fail.")
        # We'll configure it anyway, maybe it picks up from default credentials
        genai.configure()
    else:
        genai.configure(api_key=api_key)

def generate_medical_report(prompt: str, image_paths: list):
    """
    Calls Gemini Pro Vision to predict ANA pattern based on the text prompt and retrieved heatmaps.
    """
    init_gemini()
    
    # We use gemini-1.5-pro or gemini-pro-vision depending on availability
    # gemini-1.5-pro handles multimodal inputs well.
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        print(f"Failed to load gemini-1.5-pro, falling back to gemini-pro-vision. Error: {e}")
        model = genai.GenerativeModel('gemini-pro-vision')
    
    # Load images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Could not load image {path}: {e}")
            
    # Combine prompt text and images
    contents = [prompt] + images
    
    print("Calling Gemini API for medical reasoning...")
    try:
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error communicating with Gemini: {e}"

if __name__ == "__main__":
    # Simple test
    print("Testing Gemini integration...")
    res = generate_medical_report("Hello, doctor. Are you ready?", [])
    print("Response:", res)
