import cv2

def apply_privacy(frame, face_rects, enable_blur=False):
    if not enable_blur or not face_rects:
        return frame
    for (x, y, w, h) in face_rects:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi = cv2.GaussianBlur(roi, (31, 31), 0)
        frame[y:y+h, x:x+w] = roi
    return frame