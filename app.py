import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# حل مشكلة التوافق مع النماذج القديمة وتجاهل الأخطاء غير الضرورية
import keras
if hasattr(keras_layers := keras.layers, 'DepthwiseConv2D'):
    orig_init = keras_layers.DepthwiseConv2D.__init__
    def new_init(self, *args, **kwargs):
        kwargs.pop('groups', None)
        orig_init(self, *args, **kwargs)
    keras_layers.DepthwiseConv2D.__init__ = new_init

st.set_page_config(page_title="كاشف الطفيليات", layout="centered")
st.title("🔬 نظام التمييز الآلي للطفيليات")

def find_files():
    m_file = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l_file = next((f for f in os.listdir() if f.endswith(".txt") and f != "requirements.txt"), None)
    return m_file, l_file

model_path, label_path = find_files()

@st.cache_resource
def load_my_model(m_path, l_path):
    model = tf.keras.models.load_model(m_path, compile=False)
    with open(l_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

if model_path and label_path:
    try:
        model, class_names = load_my_model(model_path, label_path)
        source = st.camera_input("صوّر العينة من المجهر")
        
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="تم التقاط الصورة", use_container_width=True)
            
            # معالجة الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # --- التعديل السحري هنا ---
            # نقوم بإرسال البيانات بشكل صريح كـ Tensor واحد لتجنب خطأ "2 input tensors"
            prediction = model(tf.constant(data), training=False)
            prediction = prediction.numpy()
            # --------------------------
            
            index = np.argmax(prediction)
            label = class_names[index]
            confidence = prediction[0][index]
            
            st.success(f"النتيجة: {label}")
            st.info(f"نسبة التأكد: {confidence*100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ فني: {e}")
else:
    st.warning("يرجى رفع ملفات .h5 و .txt")
