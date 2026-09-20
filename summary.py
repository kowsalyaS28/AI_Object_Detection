import csv


# ==========================================
# 1. OPEN CSV FILE
# ==========================================

file = open("detections.csv", "r")

reader = csv.DictReader(file)


# ==========================================
# 2. STORE TRACKING IDs
# ==========================================

objects = {}


# ==========================================
# 3. READ EVERY DETECTION
# ==========================================

for row in reader:

    object_name = row["Object"]
    tracking_id = row["Tracking_ID"]

    # Ignore detections without tracking ID
    if tracking_id == "-1":
        continue

    # Create object category if not already present
    if object_name not in objects:
        objects[object_name] = set()

    # Store unique tracking ID
    objects[object_name].add(tracking_id)


# ==========================================
# 4. CLOSE FILE
# ==========================================

file.close()


# ==========================================
# 5. DISPLAY SUMMARY
# ==========================================

print()
print("========================================")
print("          DETECTION SUMMARY")
print("========================================")
print()


total_objects = 0


for object_name, tracking_ids in objects.items():

    count = len(tracking_ids)

    total_objects += count

    print(f"{object_name}: {count}")


print()
print("----------------------------------------")
print(f"Total Unique Objects: {total_objects}")
print("----------------------------------------")
print()
print("Summary generated successfully!")