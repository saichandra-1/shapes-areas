import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

st.title("🔺 Shape & Contour Analyzer")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

def detect_shape(contour):
    shape = "Unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        shape = "Quadrilateral"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices > 5:
        shape = "Circle"

    return shape, vertices

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:
            shape, vertices = detect_shape(contour)
            perimeter = cv2.arcLength(contour, True)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            cv2.drawContours(image, [contour], -1, (0,255,0), 2)
            cv2.putText(image, shape, (cx-40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            data.append([i+1, shape, vertices, round(area,2), round(perimeter,2)])

    df = pd.DataFrame(data, columns=["Object ID", "Shape", "Vertices", "Area", "Perimeter"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Analysis Results")
        st.dataframe(df)

    st.success(f"Total Objects Detected: {len(df)}")
