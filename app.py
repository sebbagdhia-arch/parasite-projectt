import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="كاشف الطفيليات", layout="centered")

st.title("🔬 نظام التمييز الآلي للطفيليات")

# دالة للبحث عن ملف النموذج تلقائياً
def find_files():
    model_file = None
    label_file = None
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
        if file.endswith(".txt") and file != "requirements.txt":
            label_file = file
    return model_file, label_file

model_path, label_path = find_files()

@st.cache_resource
def load_my_model(m_path, l_path):
    model = tf.keras.models.load_model(m_path, compile=False)
    with open(l_path, "r") as f:
        class_names = f.readlines()
    return model, class_names

if model_path and label_path:
    model, class_names = load_my_model(model_path, label_path)
    
    source = st.camera_input("صوّر العينة من المجهر")
    if source:
        image = Image.open(source).convert("RGB")
        st.image(image, caption="تم التقاط الصورة", use_container_width=True)
        
        # المعالجة والتوقع
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = img_array
        
        prediction = model.predict(data)
        index = np.argmax(prediction)
        st.success(f"النتيجة: {class_names[index][2:]}")
        st.write(f"الدقة: {prediction[0][index]*100:.2f}%")
else:
    st.error("لم نجد ملفات النموذج. تأكد من وجود ملف ينتهي بـ .h5 وملف .txt في حسابك.")
