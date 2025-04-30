import cv2
import numpy as np
import base64
import io
import fitz
from PIL import Image
import google.generativeai as genai

# === Gemini Config ===
##genai.configure(api_key="") #add your API

def remove_shadow_opencv(image_path):
    img = cv2.imread(image_path)
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_planes.append(norm)
    result = cv2.merge(result_planes)
    return Image.fromarray(result)

def convert_pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def pil_to_bytes(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return buffer.getvalue()

def extract_text_with_gemini(image_bytes):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        contents=[{
            "role": "user",
            "parts": [
                {"text": "Extract the text exactly as shown in the image without skipping."},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
            ]
        }]
    )
    return response.text.strip()
