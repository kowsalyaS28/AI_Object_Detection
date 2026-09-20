from ultralytics import YOLO
import cv2
import csv
import time
import os
from datetime import datetime


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# 2. CREATE OUTPUT FOLDER
# ==========================================

if not os.path.exists("output"):
    os.makedirs("output")


# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera opened successfully")


# ==========================================
# 4. CAMERA SETTINGS
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_camera = cap.get(cv2.CAP_PROP_FPS)

if fps_camera <= 0:
    fps_camera = 20.0


# ==========================================
# 5. VIDEO RECORDING
# ==========================================

video_path = "output/tracked_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    video_path,
    fourcc,
    fps_camera,
    (width, height)
)


# ==========================================
# 6. CSV DATA LOGGING
# ==========================================

csv_file = open(
    "detections.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Time",
    "Object",
    "Confidence",
    "Tracking_ID"
])


# ==========================================
# 7. CONFIDENCE THRESHOLD
# ==========================================

CONFIDENCE_THRESHOLD = 0.50


# ==========================================
# 8. FPS VARIABLES
# ==========================================

previous_time = 0


print("Detection started")
print("Press Q or ESC to stop")


# ==========================================
# 9. MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # Read camera frame
    # --------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break


    # --------------------------------------
    # YOLO OBJECT TRACKING
    # --------------------------------------

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # --------------------------------------
    # Draw bounding boxes and tracking IDs
    # --------------------------------------

    annotated_frame = results[0].plot()


    # ======================================
    # 10. OBJECT COUNTING
    # ======================================

    object_count = {}


    # ======================================
    # 11. PROCESS EVERY DETECTED OBJECT
    # ======================================

    for box in results[0].boxes:

        # Object class ID
        class_id = int(box.cls[0])

        # Object name
        class_name = model.names[class_id]

        # Confidence score
        confidence = float(box.conf[0])


        # ----------------------------------
        # Tracking ID
        # ----------------------------------

        if box.id is not None:
            tracking_id = int(box.id[0])
        else:
            tracking_id = -1


        # ----------------------------------
        # Count objects
        # ----------------------------------

        if class_name in object_count:
            object_count[class_name] += 1
        else:
            object_count[class_name] = 1


        # ----------------------------------
        # Get bounding box coordinates
        # ----------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # ----------------------------------
        # Display confidence + ID
        # ----------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f} "
            f"ID:{tracking_id}"
        )


        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # ==================================
        # SAVE DATA TO CSV
        # ==================================

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        csv_writer.writerow([
            current_time,
            class_name,
            round(confidence, 2),
            tracking_id
        ])


    # ======================================
    # 12. TOTAL OBJECTS
    # ======================================

    total_objects = sum(
        object_count.values()
    )


    # ======================================
    # 13. DISPLAY OBJECT COUNTS
    # ======================================

    y = 30

    for name, count in object_count.items():

        text = f"{name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30


    # --------------------------------------
    # Display total objects
    # --------------------------------------

    cv2.putText(
        annotated_frame,
        f"Total Objects: {total_objects}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ======================================
    # 14. CALCULATE FPS
    # ======================================

    current_time = time.time()

    if previous_time != 0:

        fps = 1 / (
            current_time - previous_time
        )

    else:

        fps = 0

    previous_time = current_time


    # --------------------------------------
    # Display FPS
    # --------------------------------------

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ======================================
    # 15. SAVE VIDEO FRAME
    # ======================================

    video_writer.write(
        annotated_frame
    )


    # ======================================
    # 16. SHOW CAMERA
    # ======================================

    cv2.imshow(
        "AI Object Detection - Tracking & Counting",
        annotated_frame
    )


    # ======================================
    # 17. STOP PROGRAM
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if (
        key == ord("q")
        or key == ord("Q")
        or key == 27
    ):

        print("Stopping...")
        break


# ==========================================
# 18. RELEASE EVERYTHING
# ==========================================

cap.release()

video_writer.release()

csv_file.close()

cv2.destroyAllWindows()


# ==========================================
# 19. FINAL MESSAGE
# ==========================================

print()
print("================================")
print("      PROJECT COMPLETED")
print("================================")
print()
print("Tracked video:")
print(video_path)
print()
print("Detection data:")
print("detections.csv")
print()
print("Program finished!")