import os
import requests

# If using Ollama local server
API_URL = "http://localhost:11434/api/generate"

def send_image_and_prompt(image_path: str, prompt: str, model: str = "qwen3-vl:2b"):
    # Read image and encode as base64
    import base64
    with open(image_path, "rb") as f:
        data = f.read()
    image_b64 = base64.b64encode(data).decode("utf-8")

    payload = {
        "model": model,
        # For multimodal: include prompt and base64 image
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    result = resp.json()
    # The returned structure may vary — adjust based on actual API output
    return result.get("response") or result.get("message", {}).get("content")

if __name__ == "__main__":
    img_path = "./testing-images/testing-7.jpg"
    user_prompt = "Does the animal in the images correspond to the caption on them? Look at the image carefully before giving an answer"
    out = send_image_and_prompt(img_path, user_prompt)
    print("Model output:", out)
