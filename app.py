import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Streamlit app title and description
st.title("Shape & Contour Analyzer")
st.write("Upload an image to detect geometric shapes, count objects, and compute area/perimeter.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using PIL and convert to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Display original image
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create binary image (adjust threshold as needed)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy image for drawing
    img_contour = img_cv.copy()

    # Initialize lists for shapes, areas, perimeters
    shape_counts = {"Triangle": 0, "Square/Rectangle": 0, "Pentagon": 0, "Hexagon": 0, "Circle": 0, "Unknown": 0}
    areas = []
    perimeters = []

    # Process each contour
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), closed=True)
        sides = len(approx)

        # Determine shape based on number of sides
        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            shape = "Square/Rectangle"
        elif sides == 5:
            shape = "Pentagon"
        elif sides == 6:
            shape = "Hexagon"
        elif sides > 6:
            shape = "Circle"  # Assuming high sides approximate a circle
        else:
            shape = "Unknown"

        # Update count
        if shape in shape_counts:
            shape_counts[shape] += 1
        else:
            shape_counts["Unknown"] += 1

        # Compute area and perimeter
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        areas.append(area)
        perimeters.append(perimeter)

        # Draw contour and label shape
        cv2.drawContours(img_contour, [cnt], 0, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contour, shape, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display annotated image
    st.subheader("Annotated Image with Contours and Shapes")
    img_contour_rgb = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    st.image(img_contour_rgb, use_column_width=True)

    # Display object counts
    st.subheader("Object Counts")
    total_objects = len(contours)
    st.write(f"Total Objects Detected: {total_objects}")
    for shape, count in shape_counts.items():
        st.write(f"{shape}: {count}")

    # Display areas and perimeters in a table
    st.subheader("Areas and Perimeters")
    if total_objects > 0:
        data = {"Object ID": range(1, total_objects + 1), "Area (pixels)": areas, "Perimeter (pixels)": perimeters}
        st.table(data)
    else:
        st.write("No shapes detected.")

else:
    st.write("Please upload an image to analyze.")
