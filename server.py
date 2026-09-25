# PC Server: Image Receive + Fish Counting + Count API
# --------------------------------------------------------
# Install karein: pip install flask opencv-python numpy
# Run karein: python server.py
#
# ESP32-CAM isko /upload par image bhejega
# Doosra ESP32 (OLED wala) /count se latest count mangwayega

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

latest_count = 0
SAVE_FOLDER = "received_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)


def count_fish(image_path):
    """
    Simple contour-based counting.
    NOTE: Yeh basic approach hai — accuracy lighting, background contrast,
    aur fish overlap par depend karti hai. Behtar accuracy ke liye
    trained AI model (YOLO) use karna chahiye — abhi ke liye yeh
    baseline hai jisse system chal jaye.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding - automatically best threshold dhoondta hai
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Chhote noise contours hata dein (area filter) - apni fish size ke
    # hisaab se MIN_AREA/MAX_AREA adjust karein
    MIN_AREA = 200
    MAX_AREA = 50000
    valid_contours = [c for c in contours if MIN_AREA < cv2.contourArea(c) < MAX_AREA]

    return len(valid_contours)


@app.route("/upload", methods=["POST"])
def upload():
    global latest_count

    image_data = request.data
    image_path = os.path.join(SAVE_FOLDER, "latest.jpg")
    with open(image_path, "wb") as f:
        f.write(image_data)

    try:
        latest_count = count_fish(image_path)
        print(f"Image mili. Fish count: {latest_count}")
    except Exception as e:
        print(f"Counting error: {e}")

    return jsonify({"status": "ok", "count": latest_count})


@app.route("/count", methods=["GET"])
def get_count():
    return jsonify({"count": latest_count})


if __name__ == "__main__":
    # PC ka WiFi IP address terminal mein "ipconfig" (Windows) se check karein
    # aur ESP32 sketches mein wahi IP daalein
    app.run(host="0.0.0.0", port=5000)
