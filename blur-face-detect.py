from ultralytics import YOLO
import cv2

# Load YOLO face detection model
model = YOLO("./yolov8n-face.pt")

# Input image
image_path = "./image.jpeg"

# Detect faces
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640
)

result = results[0]

# Load original image
image = cv2.imread(image_path)

# Pixelate detected faces
for box in result.boxes:

    # Get face coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Make sure coordinates stay inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Extract face
    face = image[y1:y2, x1:x2]

    if face.size == 0:
        continue

    # Pixelation ---

    # Shrink face dramatically
    small = cv2.resize(
        face,
        (12, 12),
        interpolation=cv2.INTER_LINEAR
    )

    # Scale it back using nearest-neighbor
    pixelated = cv2.resize(
        small,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST
    )

    # Replace original face
    image[y1:y2, x1:x2] = pixelated


# Save as JPEG
output_path = "./blur_result.jpeg"

cv2.imwrite(
    output_path,
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

print(f"Saved: {output_path}")

# Display SAVED JPEG
saved_image = cv2.imread(output_path)

cv2.imshow(
    "Pixelated Face Detection",
    saved_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()