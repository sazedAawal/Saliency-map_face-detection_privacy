import cv2
import numpy as np

image = cv2.imread('./blur_result.jpeg') # Input image path

# 1. SALIENCY
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
_, saliency_map = saliency.computeSaliency(image)
saliency_map = (saliency_map * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

# 2. FACE DETECTION
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# If cascade not found, download it
if face_cascade.empty():
    import urllib.request
    cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(cascade_url, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

# 3. PIXELATE FACE
img_pixelated = image.copy()
for (x, y, w, h) in faces:
    # Extract face region
    face_roi = img_pixelated[y:y+h, x:x+w].copy()
    # Pixelate
    face_roi = cv2.resize(face_roi, (w//15, h//15), interpolation=cv2.INTER_LINEAR)
    face_roi = cv2.resize(face_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    # Put back
    img_pixelated[y:y+h, x:x+w] = face_roi

print(f"Faces detected: {len(faces)}")

# BLEND & SAVE
overlay = cv2.addWeighted(img_pixelated, 0.6, heatmap, 0.4, 0)
cv2.imwrite('final_result.jpg', overlay) # Save Output image in same folder