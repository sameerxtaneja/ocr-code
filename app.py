import streamlit as st
import io
import fitz
import cv2
import numpy as np
from PIL import Image
import openai
import google.generativeai as genai
import base64

# === CONFIG 
##openai.api_key = "" #Add ur API

##genai.configure(api_key="")   #Add ur API

# === UTILS 

def remove_shadow_opencv(pil_img):
    img = np.array(pil_img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

def pdf_to_image(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()

def extract_text_with_gemini(jpeg_bytes):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        {
            "role": "user",
            "parts": [
                {"text": "Extract the handwritten or typed text exactly as shown in the image."},
                {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes}}
            ]
        }
    ])
    return response.text.strip()

def grade_with_gpt4(question, reference, student, max_marks):
    prompt = f"""
You are a CBSE Class 12 Economics teacher.

Question: {question}

Reference Answer:
{reference}

Student Answer:
{student}

Evaluate the student's answer out of {max_marks} marks and provide 1-line feedback.

Respond in the following format:
Marks: x/{max_marks}
Feedback: <your feedback>
    """
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

#  STREAMLIT UI

st.title("📚 AI Grader – CBSE Economics")
st.markdown("Upload the reference answer (PDF or image) and student's handwritten answer image to get CBSE-style evaluation.")

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader(" Upload Reference Answer (PDF/Image)", type=["pdf", "jpg", "jpeg", "png"])
with col2:
    stu_file = st.file_uploader(" Upload Student Answer (Image)", type=["jpg", "jpeg", "png"])

question = st.text_area(" Enter the Economics Question")
max_marks = st.number_input(" Maximum Marks", min_value=1, max_value=20, value=5)

if st.button(" Grade Answer") and ref_file and stu_file and question:
    with st.spinner("Processing..."):
        # --- Reference OCR
        ref_img = pdf_to_image(ref_file) if ref_file.name.endswith(".pdf") else Image.open(ref_file)
        ref_img_clean = remove_shadow_opencv(ref_img)
        ref_bytes = pil_to_bytes(ref_img_clean)
        ref_text = extract_text_with_gemini(ref_bytes)

        # --- Student OCR
        stu_img = Image.open(stu_file)
        stu_img_clean = remove_shadow_opencv(stu_img)
        stu_bytes = pil_to_bytes(stu_img_clean)
        stu_text = extract_text_with_gemini(stu_bytes)

        # --- Grading
        result = grade_with_gpt4(question, ref_text, stu_text, max_marks)

    # === OUTPUT
    st.success(" Grading Complete!")
    st.markdown("###  Reference Answer (Extracted):")
    st.text(ref_text)

    st.markdown("###  Student Answer (Extracted):")
    st.text(stu_text)

    st.markdown("###  Final Evaluation")
    st.code(result, language="markdown")
