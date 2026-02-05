import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# إعداد الصفحة لتظهر بشكل احترافي
st.set_page_config(page_title="مختبر التشخيص المجهري", page_icon="🔬")
st.title("🔬 نظام التمييز الآلي للطفيليات")
st.write("الآن يعمل على بيئة Python 3.10 المستقرة")

# وظيفة تحميل النموذج والأسماء
@st.cache_resource
def load_model_data():
    m_file = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l_file = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f.lower()), None)
    
    if m_file and l_file:
        model = tf.keras.models.load_model(m_file, compile=False)
        with open(l_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    return None, None

model, class_names = load_model_data()

if model:
    # فتح الكاميرا
    source = st.camera_input("التقط صورة العينة من المجهر")
    
    if source:
        # 1. تجهيز الصورة (Preprocessing)
        image = Image.open(source).convert("RGB")
        st.image(image, caption="الصورة المجهرية", use_container_width=True)
        
        # تحويل الصورة للمقاس المطلوب (224x224)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # تحويل الصورة إلى مصفوفة بيانات (Array)
        image_array = np.asarray(image).astype(np.float32)
        # تطبيع البيانات (Normalization)
        normalized_image_array = (image_array / 127.5) - 1
        
        # وضع الصورة في وعاء (Batch) مناسب للنموذج
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # 2. عملية التوقع (Prediction)
        with st.spinner('جاري التحليل...'):
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
        # 3. عرض النتائج النهائية
        st.divider()
        result_text = class_names[index]
        # تنظيف النص إذا كان يحتوي على أرقام في البداية
        final_name = result_text.split(" ", 1)[1] if " " in result_text else result_text
        
        st.success(f"النتيجة المتوقعة: **{final_name}**")
        st.progress(float(confidence))
        st.write(f"نسبة التأكد: {confidence * 100:.2f}%")
        
        if confidence > 0.8:
            st.balloons()
else:
    st.error("خطأ: لم يتم العثور على ملف النموذج (.h5) في GitHub!")
