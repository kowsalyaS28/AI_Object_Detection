from ultralytics import YOLO
import cv2
import time
import os


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# 2. OPEN INPUT VIDEO
# ==========================================

video_path = "output/tracked_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not be opened")
    exit()

print("Video opened successfully")


# ==========================================
# 3. GET VIDEO INFORMATION
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 20.0


# ==========================================
# 4. CREATE OUTPUT VIDEO
# ==========================================

output_path = "output/video_detection_result.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps_video,
    (width, height)
)


# ==========================================
# 5. SETTINGS
# ==========================================

CONFIDENCE_THRESHOLD = 0.50

previous_time = 0


print("Processing video...")
print("Press Q or ESC to stop")


# ==========================================
# 6. PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break


    # ======================================
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # ======================================
    # DRAW DETECTIONS
    # ======================================

    annotated_frame = results[0].plot()


    # ======================================
    # OBJECT COUNTING
    # ======================================

    object_count = {}


    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])


        # Tracking ID

        if box.id is not None:
            tracking_id = int(box.id[0])
        else:
            tracking_id = -1


        # Count object

        if class_name in object_count:
            object_count[class_name] += 1
        else:
            object_count[class_name] = 1


        # ==================================
        # DISPLAY LABEL
        # ==================================

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

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


    # ======================================
    # TOTAL OBJECTS
    # ======================================

    total_objects = sum(
        object_count.values()
    )


    # ======================================
    # DISPLAY COUNTS
    # ======================================

    y = 30

    for name, count in object_count.items():

        cv2.putText(
            annotated_frame,
            f"{name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30


    # ======================================
    # DISPLAY TOTAL
    # ======================================

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
    # CALCULATE FPS
    # ======================================

    current_time = time.time()

    if previous_time != 0:
        fps = 1 / (current_time - previous_time)
    else:
        fps = 0

    previous_time = current_time


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
    # SAVE PROCESSED FRAME
    # ======================================

    video_writer.write(annotated_frame)


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(
        "AI Video Detection",
        annotated_frame
    )


    # ======================================
    # STOP
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        print("Stopping...")
        break


# ==========================================
# 7. RELEASE EVERYTHING
# ==========================================

cap.release()

video_writer.release()

cv2.destroyAllWindows()


# ==========================================
# 8. FINAL MESSAGE
# ==========================================

print()
print("========================================")
print("       VIDEO PROCESSING COMPLETED")
print("========================================")
print()
print("Processed video:")
print(output_path)
print()
print("Program finished!")