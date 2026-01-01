"""
Toronto 311 Image Redactor - Production Build
Face detection: MediaPipe
Vehicle/plate detection: YOLOv8 via ONNX Runtime
Auth: JWT Bearer Flow
"""

import base64
import os
import time
from io import BytesIO

import cv2
import jwt
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import mediapipe as mp

app = Flask(__name__, static_folder="static", static_url_path="")

# --- Salesforce JWT Auth ---
SF_CONSUMER_KEY = os.environ.get('SF_CONSUMER_KEY')
SF_USERNAME = os.environ.get('SF_USERNAME')
SF_LOGIN_URL = os.environ.get('SF_LOGIN_URL', 'https://login.salesforce.com')
SF_PRIVATE_KEY = os.environ.get('SF_PRIVATE_KEY', '').replace('\\n', '\n')

_sf_token_cache = {'token': None, 'instance_url': None, 'expires_at': 0}

def get_sf_access_token():
    """Get Salesforce access token via JWT Bearer flow, with caching."""
    global _sf_token_cache
    
    # Return cached token if still valid (with 5 min buffer)
    if _sf_token_cache['token'] and time.time() < _sf_token_cache['expires_at'] - 300:
        return _sf_token_cache['token'], _sf_token_cache['instance_url']
    
    # Build JWT
    claim = {
        'iss': SF_CONSUMER_KEY,
        'sub': SF_USERNAME,
        'aud': SF_LOGIN_URL,
        'exp': int(time.time()) + 300
    }
    
    assertion = jwt.encode(claim, SF_PRIVATE_KEY, algorithm='RS256')
    
    # Token endpoint
    token_url = f"{SF_LOGIN_URL}/services/oauth2/token"
    resp = requests.post(token_url, data={
        'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion': assertion
    })
    
    if resp.status_code != 200:
        raise Exception(f"JWT auth failed: {resp.text}")
    
    data = resp.json()
    _sf_token_cache = {
        'token': data['access_token'],
        'instance_url': data['instance_url'],
        'expires_at': time.time() + 7200  # 2 hour cache
    }
    
    print(f"✓ Got new Salesforce token")
    return _sf_token_cache['token'], _sf_token_cache['instance_url']
def upload_image_to_salesforce(image_base64, filename="311_photo.jpg"):
    """Upload image to Salesforce as ContentDocument, return ContentDocumentId."""
    access_token, instance_url = get_sf_access_token()
    
    url = f"{instance_url}/services/data/v59.0/sobjects/ContentVersion"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'Title': f'311_Report_{int(time.time())}',
        'PathOnClient': filename,
        'VersionData': image_base64
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    
    if resp.status_code not in [200, 201]:
        print(f"ContentVersion upload failed: {resp.text}")
        return None
    
    content_version_id = resp.json().get('id')
    print(f"✓ Created ContentVersion: {content_version_id}")
    
    query_url = f"{instance_url}/services/data/v59.0/query"
    query = f"SELECT ContentDocumentId FROM ContentVersion WHERE Id = '{content_version_id}'"
    
    resp = requests.get(query_url, headers={'Authorization': f'Bearer {access_token}'}, params={'q': query})
    
    if resp.status_code == 200 and resp.json().get('records'):
        content_doc_id = resp.json()['records'][0]['ContentDocumentId']
        print(f"✓ ContentDocumentId: {content_doc_id}")
        return content_doc_id
    
    return None


def invoke_analyze_photo_flow(content_doc_id):
    """Call the Analyze 311 Photo Flow via Salesforce REST API."""
    access_token, instance_url = get_sf_access_token()
    
    url = f"{instance_url}/services/data/v62.0/actions/custom/flow/Analyze_311_Photo_Flow"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'inputs': [{
            'ContentDocumentId': content_doc_id
        }]
    }
    
    print(f'🔍 Calling Analyze 311 Photo Flow for {content_doc_id}...')
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if resp.status_code == 200:
        result = resp.json()
        if result and len(result) > 0 and 'outputValues' in result[0]:
            analysis = result[0]['outputValues'].get('AnalysisResult', '')
            print(f'✓ Flow returned analysis')
            return analysis
        print(f'⚠ Flow returned unexpected format: {result}')
        return None
    
    print(f'⚠ Flow call failed: {resp.status_code} - {resp.text}')
    return None


# --- Model Loading ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
PLATE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "plate_detect.onnx")
ort_session = None
plate_session = None

def load_models():
    global ort_session, plate_session
    if os.path.exists(MODEL_PATH):
        ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        print(f"✓ YOLOv8 vehicle model loaded")
    else:
        print(f"⚠ Vehicle model not found at {MODEL_PATH}")
    
    if os.path.exists(PLATE_MODEL_PATH):
        plate_session = ort.InferenceSession(PLATE_MODEL_PATH, providers=["CPUExecutionProvider"])
        print(f"✓ License plate model loaded")
    else:
        print(f"⚠ Plate model not found at {PLATE_MODEL_PATH}")

load_models()

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def preprocess_for_yolo(image_bgr, input_size=640):
    h, w = image_bgr.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_x, pad_y


def postprocess_yolo(output, scale, pad_x, pad_y, orig_h, orig_w, conf_thresh=0.25, iou_thresh=0.45):
    predictions = output[0].transpose()
    boxes, scores, class_ids = [], [], []
    
    for pred in predictions:
        x_c, y_c, w, h = pred[:4]
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence < conf_thresh or class_id not in VEHICLE_CLASSES:
            continue
        
        x1 = (x_c - w / 2 - pad_x) / scale
        y1 = (y_c - h / 2 - pad_y) / scale
        x2 = (x_c + w / 2 - pad_x) / scale
        y2 = (y_c + h / 2 - pad_y) / scale
        
        x1, y1 = max(0, min(x1, orig_w)), max(0, min(y1, orig_h))
        x2, y2 = max(0, min(x2, orig_w)), max(0, min(y2, orig_h))
        
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(confidence))
        class_ids.append(class_id)
    
    if not boxes:
        return []
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    results = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[idx]
        results.append({"bbox": (int(x), int(y), int(x + w), int(y + h)), "confidence": scores[idx], "class_id": class_ids[idx]})
    return results


def detect_vehicles(image_bgr):
    if ort_session is None:
        return []
    h, w = image_bgr.shape[:2]
    blob, scale, pad_x, pad_y = preprocess_for_yolo(image_bgr)
    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: blob})
    return postprocess_yolo(output[0], scale, pad_x, pad_y, h, w)


def detect_plates(image_bgr):
    if plate_session is None:
        return []
    
    h, w = image_bgr.shape[:2]
    input_size = 640
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    
    input_name = plate_session.get_inputs()[0].name
    output = plate_session.run(None, {input_name: blob})[0]
    
    plates = []
    predictions = output[0].transpose()
    
    for pred in predictions:
        x_c, y_c, pw, ph = pred[:4]
        confidence = pred[4:].max() if len(pred) > 4 else pred[4] if len(pred) == 5 else 0
        
        if confidence < 0.4:
            continue
        
        x1 = (x_c - pw / 2 - pad_x) / scale
        y1 = (y_c - ph / 2 - pad_y) / scale
        x2 = (x_c + pw / 2 - pad_x) / scale
        y2 = (y_c + ph / 2 - pad_y) / scale
        
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 > x1 and y2 > y1:
            plates.append((x1, y1, x2, y2))
    
    if len(plates) > 1:
        boxes = [[p[0], p[1], p[2]-p[0], p[3]-p[1]] for p in plates]
        scores = [1.0] * len(plates)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
        plates = [plates[i[0] if isinstance(i, (list, np.ndarray)) else i] for i in indices]
    
    return plates


def detect_faces(image_rgb):
    results = face_detector.process(image_rgb)
    faces = []
    if results.detections:
        h, w = image_rgb.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            faces.append((x1, y1, x2, y2))
    return faces


def apply_blur(image_bgr, regions, blur_strength=99):
    for (x1, y1, x2, y2) in regions:
        if x2 > x1 and y2 > y1:
            roi = image_bgr[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            image_bgr[y1:y2, x1:x2] = blurred
    return image_bgr


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "mediapipe_face": True,
            "yolov8_onnx": ort_session is not None,
            "plate_detect": plate_session is not None
        }
    })


@app.route("/redact", methods=["POST"])
def redact():
    try:
        data = request.get_json()
        if not data or "imageBase64" not in data:
            return jsonify({"error": "Missing imageBase64 field"}), 400
        
        image_data = base64.b64decode(data["imageBase64"])
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        faces = detect_faces(image_rgb)
        plates = detect_plates(image_bgr)
        vehicles = detect_vehicles(image_bgr)
        
        all_regions = faces + plates
        if all_regions:
            image_bgr = apply_blur(image_bgr, all_regions)
        
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        redacted_b64 = base64.b64encode(buffer).decode("utf-8")
        
        return jsonify({
            "redacted": len(all_regions) > 0,
            "redactedBase64": redacted_b64,
            "facesBlurred": len(faces),
            "platesBlurred": len(plates),
            "vehiclesDetected": len(vehicles)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send message to Austin agent via Apex REST with JWT auth."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('sessionId')
        image_base64 = data.get('imageBase64')
        
        # Get token via JWT Bearer
        access_token, instance_url = get_sf_access_token()
        
        # If image provided, upload to Salesforce first
        if image_base64:
            print('📷 Uploading image to Salesforce...')
            content_doc_id = upload_image_to_salesforce(image_base64)
            if content_doc_id:
                # Call the Flow directly to analyze the photo
                analysis = invoke_analyze_photo_flow(content_doc_id)
                if analysis:
                    message = f"{message}\n\n[Photo Analysis Result]\n{analysis}\n\nContentDocumentId: {content_doc_id}"
                    print('✓ Image analyzed, added to message context')
                else:
                    message = f"{message}\n\n[Photo uploaded but analysis failed. ContentDocumentId: {content_doc_id}]"
                    print('⚠ Image uploaded but flow analysis failed')
            else:
                print('⚠ Image upload failed, continuing without image')
        
        url = f"{instance_url}/services/apexrest/austin"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            'message': message,
            'sessionId': session_id
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, str):
                import json
                result = json.loads(result)
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': f'Salesforce error: {response.status_code} - {response.text}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route("/api/test-auth", methods=["GET"])
def test_auth():
    """Test JWT authentication."""
    try:
        token, instance_url = get_sf_access_token()
        return jsonify({
            'success': True,
            'instance_url': instance_url,
            'token_preview': token[:20] + '...'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
