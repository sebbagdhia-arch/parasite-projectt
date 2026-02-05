import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# -----------------------------------------------------------
# 1. تصحيح مشكلة DepthwiseConv2D (للنماذج القديمة)
# -----------------------------------------------------------
import keras
if hasattr(keras.layers, 'DepthwiseConv2D'):
    orig_init = keras.layers.DepthwiseConv2D.__init__
    def new_init(self, *args, **kwargs):
        kwargs.pop('groups', None)
        orig_init(self, *args, **kwargs)
    keras.layers.DepthwiseConv2D.__init__ = new_init

# -----------------------------------------------------------
# 2. إعدادات الصفحة
# -----------------------------------------------------------
st.set_page_config(page_title="كاشف الطفيليات", layout="centered")
st.title("🔬 نظام التمييز الآلي للطفيليات")

# -----------------------------------------------------------
# 3. تحميل النموذج
# -----------------------------------------------------------
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
    # فحص حجم الملف
    if os.path.getsize(model_path) / (1024 * 1024) < 1:
        st.error("⚠️ ملف النموذج يبدو تالفاً (أقل من 1 ميجابايت). أعد رفعه.")
        st.stop()

    try:
        model, class_names = load_my_model(model_path, label_path)
        
        source = st.camera_input("التقط صورة للعينة")
        
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="تم الالتقاط", use_container_width=True)
            
            # تجهيز الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # -----------------------------------------------------------
            # 4. التوقع الذكي (الحل للمشكلة الحالية)
            # -----------------------------------------------------------
            try:
                # المحاولة الأولى: الطريقة العادية
                prediction = model.predict(data)
            except Exception:
                # المحاولة الثانية: كسر الغلاف واستخدام الطبقة الداخلية مباشرة
                # هذا يتخطى خطأ "2 input tensors"
                prediction = model.layers[0](tf.constant(data), training=False)
                prediction = prediction.numpy()
            
            # عرض النتائج
            index = np.argmax(prediction)
            label_text = class_names[index]
            confidence = prediction[0][index]
            
            # تنظيف النص
            if " " in label_text:
                label_text = label_text.split(" ", 1)[1]
            
            st.success(f"النتيجة: **{label_text}**")
            st.metric("درجة الثقة", f"{confidence*100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
        
else:
    st.warning("يرجى رفع ملفات .h5 و .txt")
