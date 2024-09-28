import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
from streamlit_lottie import st_lottie
import requests

# ฟังก์ชันเพื่อเรียกใช้แอนิเมชัน Lottie
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ฟังก์ชันเพื่อรับค่าพื้นฐานจากภาพ
def extract_dominant_color(image, k=4):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((-1, 3))  # Flatten the image to a 2D array (each pixel as a row)

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(image)

        colors = kmeans.cluster_centers_.astype(int)
        return colors[0]  # Return the most frequent color
    except Exception as e:
        st.error(f"ไม่สามารถประมวลผลภาพได้: {str(e)}")
        return None

# ฟังก์ชันสำหรับการกำหนดโทนสีตามฤดูกาล
from scipy.spatial import distance

# ข้อมูล RGB ของแต่ละ Personal Color
personal_colors = {
    'Spring': [(253, 213, 177), (255, 224, 181), (255, 212, 160), (244, 168, 131),
               (253, 210, 156), (255, 193, 137), (255, 215, 178), (251, 184, 147),
               (255, 222, 164), (246, 178, 140)],
    'Summer': [(240, 196, 179), (232, 177, 167), (244, 228, 214), (220, 196, 187),
               (247, 222, 216), (240, 192, 190), (242, 206, 201), (234, 198, 185),
               (222, 200, 192), (245, 209, 207)],
    'Autumn': [(219, 171, 128), (214, 150, 110), (225, 167, 128), (201, 135, 94),
               (216, 176, 157), (182, 123, 85), (203, 145, 104), (223, 168, 138),
               (179, 121, 94), (174, 133, 108)],
    'Winter': [(238, 217, 215), (219, 202, 197), (193, 175, 169), (156, 132, 128),
               (233, 207, 203), (195, 178, 176), (176, 165, 163), (168, 157, 152),
               (214, 191, 191), (142, 126, 123)]
}


# ฟังก์ชันเพื่อวิเคราะห์ว่าโทนสีอยู่ใน Personal Color ใด
def classify_seasonal_color(color):
    if color is None:
        return 'Cannot be verified'

    # คำนวณระยะทางเชิงยุคลิด (Euclidean distance) ระหว่างสีที่รับเข้ามากับแต่ละ Personal Color
    distances = {}
    for season, colors in personal_colors.items():
        distances[season] = min([distance.euclidean(color, c) for c in colors])

    # เลือก Personal Color ที่มีระยะทางน้อยที่สุด
    season = min(distances, key=distances.get)

    return season

# ฟังก์ชันแนะนำการแต่งตัวตามโทนสีฤดูกาล
def outfit_suggestion(season):
    suggestions = {
        'Spring': 'เลือกเสื้อผ้าที่มีเฉดสีสว่าง เพื่อให้ลุคสดใสและกระปรี้กระเปร่า',
        'Summer': 'เลือกเสื้อผ้าที่มีเฉดสีเย็นและอ่อน เพื่อให้ดูอ่อนโยนและสงบ',
        'Autumn': 'เลือกเสื้อผ้าที่มีเฉดสีอบอุ่นเข้ม เพื่อให้ลุคหรูหราและอบอุ่น',
        'Winter': 'เลือกเสื้อผ้าที่มีเฉดสีเย็นและเข้ม เพื่อให้ดูชัดเจนและทรงพลัง',
    }
    return suggestions.get(season, 'ไม่สามารถประมวลผลได้')

# ฟังก์ชันสำหรับแสดงสีตัวอย่าง
def color_suggestions_for_season(season):
    seasonal_colors = {
        'Spring': ['#fce300', '#ffb81c', '#a6631b', '#c4d600', '#74aa50', '#009f4d',
                   '#2dccd3', '#9595d2', '#963cbd', '#ec5037', '#ff8200', '#ffa38b'],
        'Summer': ['#f395c7', '#f57eb6', '#c964cf', '#00a376', '#71c5e8', '#00a9e0',
                   '#0077c8', '#93328e', '#ab145a', '#bc0f3f', '#484a51', '#003057'],
        'Autumn': ['#ffcd00', '#b89d18', '#8f993e', '#5e7e29', '#a6631b', '#9d4815',
                   '#7c4d3a', '#9a3324', '#ec5037', '#ef7200', '#00bfb3', '#00778b'],
        'Winter': ['#fefefe', '#99d6ea', '#f395c7', '#f8e59a', '#00a3e1', '#0057b8',
                   '#004b87', '#84329b', '#963cbd', '#ce0037', '#808286', '#131313']
    }
    return seasonal_colors.get(season, [])

# ฟังก์ชันสำหรับสีที่ควรหลีกเลี่ยง
def color_nonsuggestions_for_season(season):
    nonsuggestions = {
        'Spring': ['#f5a0c8', '#bb69b1', '#65688f', '#9b999e', '#2d2d4d'],
        'Summer': ['#f3ff00', '#ff0093', '#ff0000', '#3bcf46', '#402918'],
        'Autumn': ['#ffcde6', '#b5daf3', '#a3a5db', '#a7a4ab', '#a7a4ab'],
        'Winter': ['#fb9d83', '#b34e56', '#dac000', '#9ea831', '#a47841'],
    }
    return nonsuggestions.get(season, [])

# โหลด Lottie Animation
lottie_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_h4t3z0y7.json")

# เริ่มต้นแอป Streamlit
st.title("🎨 Colorista: Your Personal Color Consultant")
st.subheader("ค้นพบโทนสีที่เหมาะกับคุณพร้อมแนะนำการแต่งตัว!")

# แสดงแอนิเมชัน
if lottie_animation:
    st_lottie(lottie_animation, speed=1, height=200, key="upload")

uploaded_file = st.file_uploader("อัพโหลดภาพของคุณ", type=["jpg", "jpeg", "png"])
st.write("💡 เคล็ดลับ: กรุณาใส่รูปภาพที่เห็นเพียงสีผิวของคุณเท่านั้น หลีกเลี่ยงการแต่งหน้า")

if uploaded_file is not None:
    st.write("⏳ กำลังประมวลผลรูปภาพของคุณ...")

    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    time.sleep(2)  # เพิ่มการรอเพื่อความน่าตื่นเต้น

    st.image(image, channels="BGR", caption="ภาพต้นฉบับ")

    dominant_color = extract_dominant_color(image)

    if dominant_color is not None:
        st.write(f"🔍 สีที่เด่นคือ: {dominant_color}")
        hex_color = f"#{dominant_color[0]:02X}{dominant_color[1]:02X}{dominant_color[2]:02X}"
        st.color_picker("สีที่เด่น", value=hex_color)

        # ตรวจสอบโทนสีตามฤดูกาล
        season = classify_seasonal_color(dominant_color)

        st.write(f"✨ โทนสีของคุณคือ: {season}")
        st.write(outfit_suggestion(season))

        # แสดงตัวอย่างสีที่เหมาะสม
        st.write("👗 ตัวอย่างสีที่เหมาะสมสำหรับการแต่งตัว:")
        color_suggestions = color_suggestions_for_season(season)
        if color_suggestions:
            cols = st.columns(len(color_suggestions))
            for i, color in enumerate(color_suggestions):
                with cols[i]:
                    st.markdown(f"<div style='width:60px; height:100px; background-color:{color};'></div>", unsafe_allow_html=True)

        # แสดงสีที่ควรหลีกเลี่ยง
        st.write("🚫 สีที่ควรหลีกเลี่ยง:")
        color_nonsuggestions = color_nonsuggestions_for_season(season)
        if color_nonsuggestions:
            cols_nonsug = st.columns(len(color_nonsuggestions))
            for i, color in enumerate(color_nonsuggestions):
                with cols_nonsug[i]:
                    st.markdown(f"<div style='width:100px; height:100px; background-color:{color};'></div>", unsafe_allow_html=True)

