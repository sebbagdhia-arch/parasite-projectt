import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import keras

# 1. إصلاح الطبقات القديمة (DepthwiseConv2D)
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 نظام التشخيص الآلي (الإصدار المصحح)")

def find_files():
    m = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f), None)
    return m, l

model_path, label_path = find_files()

@st.cache_resource
def load_and_fix_model(m_path, l_path):
    custom_objects = {'DepthwiseConv2D': PatchedDepthwiseConv2D}
    # تحميل النموذج الأساسي
    base_model = tf.keras.models.load_model(m_path, custom_objects=custom_objects, compile=False)
    
    # --- العملية الجراحية ---
    # إذا كان النموذج عبارة عن "غلاف" (Sequential)، سنخترقه للوصول للمحرك الداخلي
    if hasattr(base_model, 'layers'):
        for layer in base_model.layers:
            if "functional" in layer.name.lower() or isinstance(layer, tf.keras.Model):
                final_model = layer
                break
        else:
            final_model = base_model
    else:
        final_model = base_model
        
    with open(l_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return final_model, labels

if model_path and label_path:
    try:
        model, class_names = load_and_fix_model(model_path, label_path)
        
        source = st.camera_input("صوّر العينة من المجهر")
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="الصورة الحالية", use_container_width=True)
            
            # معالجة الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # --- التوقع المباشر (تجاهل القناع تماماً) ---
            # نستخدم استدعاء الطبقة مباشرة بدون predict() لتجنب إرسال mask
            prediction = model(tf.constant(data), training=False)
            if hasattr(prediction, "numpy"):
                prediction = prediction.numpy()
            
            index = np.argmax(prediction)
            confidence = prediction[0][index] * 100
            
            st.balloons()
            st.success(f"النتيجة: {class_names[index]}")
            st.metric("دقة التشخيص", f"{confidence:.2f}%")
            
    except Exception as e:
        st.error(f"خطأ في معالجة النموذج: {e}")
else:
    st.warning("يرجى التأكد من وجود ملفات .h5 و .txt")
