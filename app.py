import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Gender Detector", page_icon="👤", layout="wide")

# Load Trained Gender Model
def load_gender_model():
    try:
        model = load_model("face_detect.keras")  # Make sure this file exists
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Preprocess the image for gender detection
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize according to model input
    img_array = np.array(img)

    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if required

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Shape: (1, 150, 150, 1)

    return img_array

# Predict gender
def predict_gender(image):
    model = load_gender_model()
    if model is None:
        return None

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    class_idx = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    class_names = ['Men', 'Women']  # Update based on your model's training

    return {
        'class': class_names[class_idx],
        'confidence': confidence * 100
    }

# Streamlit Main App
def main():
    st.title("👤 Face Gender Detection")

    input_method = st.radio("Select Input Method", ["Upload Image", "Camera Capture"])

    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_input = st.camera_input("Take a selfie")
        if camera_input:
            image = Image.open(camera_input).convert('RGB')

    if image:
        st.image(image, width=300, caption="Input Face")

        if st.button("Detect Gender"):
            with st.spinner("Analyzing..."):
                result = predict_gender(image)

                if result:
                    st.success(f"Prediction: *{result['class']}*")

# Run the app
if __name__ == "__main__":
    main()
