import streamlit as st
import pandas as pd
import os


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Object Detection Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. TITLE
# ==========================================

st.title("🤖 AI Object Detection Dashboard")

st.write(
    "Real-Time Object Detection, Tracking and Counting"
)


# ==========================================
# 3. CHECK CSV FILE
# ==========================================

if not os.path.exists("detections.csv"):

    st.error("detections.csv not found.")

    st.stop()


# ==========================================
# 4. READ CSV
# ==========================================

df = pd.read_csv("detections.csv")


# ==========================================
# 5. CLEAN DATA
# ==========================================

df = df[df["Tracking_ID"] != -1]


# ==========================================
# 6. UNIQUE OBJECTS
# ==========================================

unique_objects = df.drop_duplicates(
    subset=["Object", "Tracking_ID"]
)


# ==========================================
# 7. CALCULATE VALUES
# ==========================================

total_objects = len(unique_objects)

total_detections = len(df)

average_confidence = df["Confidence"].mean()


# ==========================================
# 8. DISPLAY KPI CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Unique Objects",
        total_objects
    )

with col2:
    st.metric(
        "Total Detections",
        total_detections
    )

with col3:
    st.metric(
        "Average Confidence",
        f"{average_confidence:.2f}"
    )


# ==========================================
# 9. OBJECT COUNTS
# ==========================================

st.subheader("📊 Object Detection Summary")


object_summary = (
    unique_objects["Object"]
    .value_counts()
    .reset_index()
)

object_summary.columns = [
    "Object",
    "Count"
]


# ==========================================
# 10. DISPLAY TABLE
# ==========================================

st.dataframe(
    object_summary,
    use_container_width=True
)


# ==========================================
# 11. BAR CHART
# ==========================================

st.subheader("📈 Object Count")

st.bar_chart(
    object_summary.set_index("Object")
)


# ==========================================
# 12. DETECTION DATA
# ==========================================

st.subheader("🔍 Detection Data")

st.dataframe(
    df,
    use_container_width=True
)


# ==========================================
# 13. VIDEO
# ==========================================

video_path = "output/video_detection_result.mp4"

if os.path.exists(video_path):

    st.subheader("🎥 Processed Video")

    video_file = open(video_path, "rb")

    video_bytes = video_file.read()

    st.video(video_bytes)

    video_file.close()

else:

    st.warning(
        "Processed video not found."
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.markdown("---")

st.write(
    "AI Object Detection System | "
    "Python • YOLO • OpenCV • Streamlit"
)