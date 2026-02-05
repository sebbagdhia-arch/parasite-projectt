import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import keras

# --- 1. حل مشكلة تحميل الطبقات القديمة ---
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 مختبر التشخيص الذكي (إصدار 2026)")

def find_files():
    m = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f), None)
    return m, l

model_path, label_path = find_files()

@st.cache_resource
def load_model_safely(m_path, l_path):
    custom_objects = {'DepthwiseConv2D': PatchedDepthwiseConv2D}
    model = tf.keras.models.load_model(m_path, custom_objects=custom_objects, compile=False)
    
    # استخراج المحرك الداخلي إذا كان النموذج مغلفاً بـ Sequential
    if isinstance(model, tf.keras.Sequential):
        model = model.layers[0]
        
    with open(l_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

if model_path and label_path:
    try:
        model, class_names = load_model_safely(model_path, label_path)
        
        source = st.camera_input("التقط صورة من المجهر")
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="العينة المراد تحليلها", use_container_width=True)
            
            # معالجة الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # --- 3. التوقع الآمن (تخطي خطأ Tensors 2) ---
            # هنا نقوم باستدعاء النموذج كدالة مباشرة لتجنب تكرار البيانات
            prediction = model(data, training=False)
            if hasattr(prediction, "numpy"):
                prediction = prediction.numpy()
                
            index = np.argmax(prediction)
            
            st.balloons()
            st.success(f"النتيجة: {class_names[index]}")
            st.metric("نسبة التأكد", f"{prediction[0][index]*100:.2f}%")
            
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
else:
    st.warning("يرجى التأكد من رفع ملفات .h5 و .txt")
