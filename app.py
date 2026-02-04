import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. إعدادات واجهة الموقع
st.set_page_config(
    page_title="كاشف الطفيليات الذكي",
    page_icon="🔬",
    layout="centered"
)

# تحسين المظهر بالعربية
st.markdown("""
    <style>
    .main { text-align: right; direction: rtl; }
    stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """, unsafe_allow_ Harris=True)

st.title("🔬 نظام التمييز الآلي للطفيليات المجهرية")
st.write("مرحباً بك يا دكتور. هذا النظام يستخدم الذكاء الاصطناعي لتحليل عينات البراز المجهرية.")
st.info("قم بالتقاط صورة للعينة من المجهر أو ارفع صورة من الجهاز وسيقوم النظام بتشخيصها.")

# 2. تحميل نموذج الذكاء الاصطناعي
@st.cache_resource
def load_my_model():
    # تحميل النموذج والعناوين
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

try:
    model, class_names = load_my_model()
except Exception as e:
    st.error("خطأ: لم يتم العثور على ملفات النموذج keras_model.h5 أو labels.txt")
    st.stop()

# 3. دالة معالجة الصورة والتوقع
def predict(image_data, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # إجراء التوقع
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:].strip(), confidence_score

# 4. خيارات الإدخال (كاميرا الهاتف أو رفع ملف)
tab1, tab2 = st.tabs(["📸 تصوير مباشر (المجهر)", "📂 اختيار صورة من الجهاز"])

with tab1:
    img_file_buffer = st.camera_input("وجه كاميرا الهاتف نحو عدسة المجهر")

with tab2:
    uploaded_file = st.file_uploader("اختر صورة مجهرية واضحة", type=["jpg", "png", "jpeg"])

# تحديد الصورة المختارة
source = img_file_buffer if img_file_buffer else uploaded_file

# 5. عرض النتائج
if source is not None:
    image = Image.open(source).convert("RGB")
    st.image(image, caption="العينة المراد فحصها", use_container_width=True)
    
    with st.spinner("جاري التحليل والمقارنة مع قاعدة البيانات..."):
        label, score = predict(image, model, class_names)
    
    st.divider()
    
    # عرض النتيجة بشكل بارز
    st.subheader(f"النتيجة المتوقعة: {label}")
    st.progress(float(score))
    st.write(f"نسبة دقة التشخيص: {score*100:.2f}%")
    
    # تفصيل إضافي بناءً على النتيجة
    if score > 0.80:
        st.success(f"تشخيص قوي: تم التعرف على {label} بوضوح.")
    elif score > 0.50:
        st.warning("تشخيص محتمل: يرجى تحسين إضاءة المجهر أو التركيز (Focus) وإعادة التصوير.")
    else:
        st.error("غير قادر على التشخيص: الصورة غير واضحة أو الطفيلي غير مدرج في قاعدة البيانات.")

st.write("---")
st.caption("مشروع تخرج طالب مخبري - تحت إشراف الذكاء الاصطناعي 2026")