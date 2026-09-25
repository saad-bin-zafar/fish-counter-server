import os
import io
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

# API key environment variable se aayegi (Hugging Face Secrets me set karenge)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# Hardcoded prompt - hamesha yehi use hoga, kabhi manually type nahi karna
FISH_COUNT_PROMPT = (
    "Count the number of fish visible in this image. "
    "Reply with ONLY a single number, nothing else. "
    "If no fish are visible, reply with 0."
)


@app.route("/", methods=["GET"])
def home():
    return "Fish Counter Server is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ESP32 se image file expect kar rahe hain (form-data, key = "image")
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read()))

        # Gemini ko image + hardcoded prompt bhejna
        response = model.generate_content([FISH_COUNT_PROMPT, image])

        text_result = response.text.strip()

        # Sirf number nikalne ki koshish (safety ke liye)
        digits = "".join(c for c in text_result if c.isdigit())
        fish_count = int(digits) if digits else 0

        return jsonify({"fish_count": fish_count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
