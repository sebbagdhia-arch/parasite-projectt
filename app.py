import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- الحل السحري لمشكلة DepthwiseConv2D ---
# نقوم بإنشاء نسخة معدلة من الطبقة تتجاهل كلمة 'groups'
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups') # حذف الكلمة المسببة للخطأ
        super().__init__(*args, **kwargs)

# إعداد الواجهة
st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 مختبر التشخيص الذكي")

def find_files():
    m = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f), None)
    return m, l

model_path, label_path = find_files()

@st.cache_resource
def load_model_safely(m_path, l_path):
    # إخبار Keras باستخدام الطبقة المعدلة بدلاً من الأصلية
    custom_objects = {'DepthwiseConv2D': PatchedDepthwiseConv2D}
    model = tf.keras.models.load_model(m_path, custom_objects=custom_objects, compile=False)
    
    with open(l_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

if model_path and label_path:
    try:
        model, class_names = load_model_safely(model_path, label_path)
        
        source = st.camera_input("التقط صورة من المجهر")
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="العينة الملتقطة", use_container_width=True)
            
            # المعالجة (نفس مقاييس Teachable Machine)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # التوقع
            prediction = model.predict(data)
            index = np.argmax(prediction)
            
            # عرض النتيجة
            st.balloons()
            st.success(f"النتيجة: {class_names[index]}")
            st.write(f"نسبة التأكد: {prediction[0][index]*100:.2f}%")
            
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
else:
    st.warning("يرجى التأكد من وجود ملفات .h5 و .txt في حسابك على GitHub")
