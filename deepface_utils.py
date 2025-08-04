from deepface import DeepFace

# Phân tích ảnh tĩnh (Ảnh mode) và full-frame video (Enhanced Video)
def analyze_face(frame):
    """
    Dùng DeepFace để detect và phân tích cảm xúc trên toàn bộ frame.
    Trả về {'region': {...}, 'emotion': dominant_emotion} hoặc None.
    """
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='retinaface',  # backend chính xác cho detection
            enforce_detection=True
        )
        if isinstance(result, list):
            result = result[0]
        return {'region': result.get('region'), 'emotion': result.get('dominant_emotion')}
    except Exception as e:
        print(f"DeepFace analyze error: {e}")
        return None

# Phân tích ROI (Webcam/Video chuyển tiếp giữa full-frame)
def predict_emotion_roi(roi):
    """
    Dùng DeepFace để phân tích cảm xúc trên ROI nhỏ.
    Trả về dominant_emotion hoặc None.
    """
    try:
        result = DeepFace.analyze(
            roi,
            actions=['emotion'],
            model_name='ArcFace',
            detector_backend='opencv',  # nhanh cho ROI
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result.get('dominant_emotion')
    except Exception as e:
        print(f"DeepFace ROI error: {e}")
        return None