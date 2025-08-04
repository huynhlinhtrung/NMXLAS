import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import deque, Counter

from deepface_utils import analyze_face, predict_emotion_roi

st.title("Nhận diện cảm xúc khuôn mặt - DeepFace")
os.makedirs('output', exist_ok=True)

# --- Shared configs ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Webcam/Video pipeline: full-frame every N, fallback ROI
FRAME_FULL_COUNT = 0
FULL_SKIP = 15       # Phân tích full-frame mỗi 15 khung
last_full = None     # {'region', 'emotion'}

FRAME_COUNTER = 0
SKIP_FRAMES = 5      # ROIs phân tích mỗi 5 khung
HISTORY_LEN = 5
emotion_history = {} # face_id -> deque
MARGIN = 0.2
ROI_SIZE = 160

mode = st.sidebar.selectbox("Chọn chế độ", ["Webcam", "Ảnh", "Video"])

# --- Webcam mode (hybrid) ---
if mode == "Webcam":
    run = st.checkbox("Bật camera", True)
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret: break

        # full-frame periodic
        FRAME_FULL_COUNT += 1
        if FRAME_FULL_COUNT % FULL_SKIP == 0:
            last_full = analyze_face(frame)
            # reset ROI history
            emotion_history.clear()

        # draw full-frame result
        if last_full and last_full['region']:
            r = last_full['region']
            emo = last_full['emotion']
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if emo:
                cv2.putText(frame, emo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # ROI fallback on intermediate frames
        FRAME_COUNTER += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for i,(x,y,fw,fh) in enumerate(faces):
            roi = frame[y:y+fh, x:x+fw]
            if roi.size==0: continue
            roi_small = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
            if i not in emotion_history:
                emotion_history[i] = deque(maxlen=HISTORY_LEN)
            if FRAME_COUNTER % SKIP_FRAMES == 0:
                e = predict_emotion_roi(roi_small)
                emotion_history[i].append(e)
            hist = [e for e in emotion_history[i] if e]
            disp = Counter(hist).most_common(1)[0][0] if hist else None
            if disp:
                cv2.putText(frame, disp, (x, y+fh+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

# --- Ảnh mode (static) ---
elif mode == "Ảnh":
    up = st.file_uploader("Tải ảnh", type=["jpg","jpeg","png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        res = analyze_face(img)
        if res and res['region']:
            r = res['region']; emo = res['emotion']
            x,y,w,h = r['x'],r['y'],r['w'],r['h']
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            if emo:
                cv2.putText(img, emo, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# --- Video mode (hybrid) ---
elif mode == "Video":
    up = st.file_uploader("Tải video", type=["mp4","avi","mov"])
    if up:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.'+up.name.split('.')[-1])
        tf.write(up.read())
        cap = cv2.VideoCapture(tf.name)
        stframe = st.empty()
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/result.avi', fourcc, fps, (vw,vh))
        FRAME_FULL_COUNT=0; FRAME_COUNTER=0; emotion_history.clear(); last_full=None
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret: break
            # apply same hybrid pipeline
            FRAME_FULL_COUNT +=1
            if FRAME_FULL_COUNT % FULL_SKIP==0:
                last_full = analyze_face(frame)
                emotion_history.clear()
            if last_full and last_full['region']:
                r=last_full['region']; emo=last_full['emotion']
                x,y,w,h = r['x'],r['y'],r['w'],r['h']
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                if emo: cv2.putText(frame,emo,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            FRAME_COUNTER+=1
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.2,5)
            for i,(x,y,fw,fh) in enumerate(faces):
                roi=frame[y:y+fh,x:x+fw];
                if roi.size==0: continue
                roi_s=cv2.resize(roi,(ROI_SIZE,ROI_SIZE))
                if i not in emotion_history: emotion_history[i]=deque(maxlen=HISTORY_LEN)
                if FRAME_COUNTER%SKIP_FRAMES==0:
                    e=predict_emotion_roi(roi_s)
                    emotion_history[i].append(e)
                hist=[e for e in emotion_history[i] if e]
                disp=Counter(hist).most_common(1)[0][0] if hist else None
                if disp: cv2.putText(frame,disp,(x,y+fh+25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            out.write(frame)
            stframe.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release(); out.release(); st.success("✅ Video processed: output/result.avi")