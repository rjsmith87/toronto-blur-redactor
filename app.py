import os
import base64
import logging
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import urllib.request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,  # 1 = full range model (better for varied distances)
    min_detection_confidence=0.5
)

# License plate model - will be downloaded on first run
plate_model = None
PLATE_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
PLATE_MODEL_PATH = "/tmp/yolov8n.pt"

def get_plate_model():
    """Load or download the YOLO model for object detection."""
    global plate_model
    if plate_model is None:
        logger.info("Loading YOLO model...")
        # Use YOLOv8n base model - we'll filter for vehicle-related objects
        # and use heuristics to find plates on detected vehicles
        plate_model = YOLO('yolov8n.pt')
        logger.info("YOLO model loaded")
    return plate_model

def blur_region(image, x1, y1, x2, y2, blur_strength=99):
    """Apply strong Gaussian blur to a region of the image."""
    # Ensure coordinates are valid integers within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    
    # Apply heavy blur for privacy
    blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 30)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def detect_faces_mediapipe(image):
    """
    Detect faces using Google MediaPipe.
    - Trained on millions of diverse faces
    - Handles all skin tones, ages, angles
    - Works in varied lighting conditions
    """
    # MediaPipe expects RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_image)
    
    faces = []
    h, w = image.shape[:2]
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # Add padding for better coverage
            pad_w = int((x2 - x1) * 0.15)
            pad_h = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            faces.append((x1, y1, x2, y2))
    
    return faces

def detect_license_plates(image):
    """
    Detect license plates using a multi-strategy approach:
    1. Use YOLO to detect vehicles (cars, trucks, motorcycles)
    2. Within vehicle bounding boxes, find plate-like regions
    3. Also scan full image for plate-like rectangles
    """
    model = get_plate_model()
    h, w = image.shape[:2]
    plates = []
    
    # Run YOLO detection
    results = model(image, verbose=False)
    
    # COCO classes for vehicles: car(2), motorcycle(3), bus(5), truck(7)
    vehicle_classes = [2, 3, 5, 7]
    vehicle_boxes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, cls in enumerate(boxes.cls):
                if int(cls) in vehicle_classes:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    vehicle_boxes.append(xyxy)
    
    # For each vehicle, look for license plate region
    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = map(int, vbox)
        vehicle_roi = image[vy1:vy2, vx1:vx2]
        
        if vehicle_roi.size == 0:
            continue
        
        # License plates are typically in the lower portion of the vehicle bbox
        vh, vw = vehicle_roi.shape[:2]
        search_region = vehicle_roi[int(vh*0.3):, :]  # Bottom 70% of vehicle
        
        plate_candidates = find_plate_candidates(search_region)
        
        for (px1, py1, px2, py2) in plate_candidates:
            # Convert back to full image coordinates
            abs_x1 = vx1 + px1
            abs_y1 = vy1 + int(vh*0.3) + py1
            abs_x2 = vx1 + px2
            abs_y2 = vy1 + int(vh*0.3) + py2
            plates.append((abs_x1, abs_y1, abs_x2, abs_y2))
    
    # Also scan full image for plates (catches parked cars, unusual angles)
    full_image_plates = find_plate_candidates(image)
    plates.extend(full_image_plates)
    
    # Remove duplicates and overlapping boxes
    plates = non_max_suppression(plates)
    
    return plates

def find_plate_candidates(image):
    """
    Find license plate candidates using computer vision:
    - Edge detection to find character-dense regions
    - Contour analysis for rectangular shapes
    - Aspect ratio filtering (plates are typically 2:1 to 5:1)
    """
    if image.size == 0:
        return []
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    candidates = []
    
    # Multi-scale approach for different plate sizes
    for scale in [1.0, 0.5, 0.25]:
        if scale != 1.0:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            scaled = gray
        
        # Preprocessing
        blurred = cv2.bilateralFilter(scaled, 11, 17, 17)
        
        # Adaptive thresholding to handle different lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 19, 9
        )
        
        # Edge detection
        edges = cv2.Canny(thresh, 50, 150)
        
        # Morphological operations to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sh, sw = scaled.shape[:2]
        
        for contour in contours:
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            
            if ch == 0:
                continue
            
            aspect_ratio = cw / ch
            area = cw * ch
            area_ratio = area / (sw * sh)
            
            # License plate aspect ratios vary by region:
            # - North American: ~2:1
            # - European: ~4.5:1 to 5:1
            # - Some square plates exist too
            if 1.5 <= aspect_ratio <= 6.0 and 0.002 <= area_ratio <= 0.15:
                # Check edge density (plates have lots of edges from characters)
                roi = edges[y:y+ch, x:x+cw]
                if roi.size > 0:
                    edge_density = cv2.countNonZero(roi) / (cw * ch)
                    
                    if edge_density > 0.15:  # Plates have at least 15% edges
                        # Scale coordinates back
                        px1 = int(x / scale)
                        py1 = int(y / scale)
                        px2 = int((x + cw) / scale)
                        py2 = int((y + ch) / scale)
                        
                        # Add padding
                        pad = int(min(cw, ch) / scale * 0.2)
                        px1 = max(0, px1 - pad)
                        py1 = max(0, py1 - pad)
                        px2 = min(w, px2 + pad)
                        py2 = min(h, py2 + pad)
                        
                        candidates.append((px1, py1, px2, py2))
    
    return candidates

def non_max_suppression(boxes, overlap_thresh=0.3):
    """Remove overlapping bounding boxes, keeping the largest."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        
        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)
        indices = indices[1:][iou < overlap_thresh]
    
    return [tuple(boxes[i].astype(int)) for i in keep]

def process_image(image_bytes):
    """Main processing function - detect and blur faces and plates."""
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    logger.info(f"Processing image: {image.shape[1]}x{image.shape[0]}")
    
    # Detect and blur faces
    faces = detect_faces_mediapipe(image)
    logger.info(f"Detected {len(faces)} faces")
    for (x1, y1, x2, y2) in faces:
        image = blur_region(image, x1, y1, x2, y2)
    
    # Detect and blur license plates
    plates = detect_license_plates(image)
    logger.info(f"Detected {len(plates)} license plates")
    for (x1, y1, x2, y2) in plates:
        image = blur_region(image, x1, y1, x2, y2)
    
    # Encode result
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes(), len(faces), len(plates)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "toronto-blur-redactor"})

@app.route('/redact', methods=['POST'])
def redact():
    try:
        data = request.get_json() or {}
        image_base64 = data.get('imageBase64')
        image_url = data.get('imageUrl')
        metadata = data.get('metadata', {})
        
        if not image_base64 and not image_url:
            return jsonify({"error": "No image provided", "redacted": False}), 400
        
        # Get image bytes
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
        else:
            return jsonify({"error": "URL fetch not implemented"}), 400
        
        # Process
        redacted_bytes, faces_count, plates_count = process_image(image_bytes)
        redacted_base64 = base64.b64encode(redacted_bytes).decode('utf-8')
        
        return jsonify({
            "redacted": True,
            "redactedBase64": redacted_base64,
            "facesBlurred": faces_count,
            "platesBlurred": plates_count,
            "metadata": metadata
        })
        
    except Exception as e:
        logger.exception("Error processing image")
        return jsonify({"error": str(e), "redacted": False}), 500

if __name__ == '__main__':
    # Pre-load models
    logger.info("Pre-loading models...")
    get_plate_model()
    logger.info("Models loaded, starting server...")
    
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
