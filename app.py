import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- حل مشكلة التوافق مع الطبقات القديمة ---
import keras
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# --- إعدادات الواجهة ---
st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 نظام التشخيص الذكي للطفيليات")
st.write("---")

# --- دالة البحث عن الملفات وتحميلها ---
@st.cache_resource
def load_everything():
    # البحث عن ملف النموذج وملف الأسماء
    model_path = next((f for f in os.listdir() if f.endswith(".h5")), None)
    label_path = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f.lower()), None)
    
    if not model_path or not label_path:
        return None, None
    
    # تحميل النموذج مع الحلول التقنية
    custom_objects = {'DepthwiseConv2D': PatchedDepthwiseConv2D}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
    # قراءة الأسماء وتنظيفها
    with open(label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return model, class_names

# تشغيل التحميل
model, class_names = load_everything()

if model and class_names:
    # واجهة الكاميرا
    img_file = st.camera_input("وجه المجهر نحو الكاميرا والتقط الصورة")
    
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="الصورة الملتقطة", use_container_width=True)
        
        # تجهيز الصورة للمعالجة
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(img_array, axis=0)
        
        # التوقع (استخدام استدعاء مباشر لتجنب أخطاء Tensors)
        with st.spinner('جاري التحليل...'):
            prediction = model(tf.constant(data), training=False).numpy()
            index = np.argmax(prediction)
            label = class_names[index]
            confidence = prediction[0][index]
            
            # تنظيف الاسم من الأرقام (مثلاً "0 Parasite" تصبح "Parasite")
            clean_label = label.split(" ", 1)[1] if " " in label else label
            
            # عرض النتائج
            st.success(f"النتيجة: **{clean_label}**")
            st.progress(float(confidence))
            st.write(f"نسبة التأكد: {confidence*100:.2f}%")
            if confidence > 0.8:
                st.balloons()
else:
    st.warning("⚠️ يرجى التأكد من رفع ملفات .h5 و .txt بشكل صحيح إلى المستودع.")
