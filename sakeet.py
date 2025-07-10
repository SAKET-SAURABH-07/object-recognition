import cv2
import numpy as np
import os
import time
import pyttsx3
import threading

FOCUS_BOX_SIZE = 200
SAVE_PATH = "captures"
PIXELS_PER_CM = 20.0
QUIT_BUTTON_POS = (540, 10, 90, 30)

os.makedirs(SAVE_PATH, exist_ok=True)
quit_requested = False

engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_shape(c):
    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
    sides = len(approx)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    if perimeter == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity > 0.85:
        return "Circle"
    
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.95 < ar < 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif sides == 7:
        return "Heptagon"
    elif sides == 8:
        return "Octagon"
    elif sides == 9:
        return "Nonagon"
    elif sides == 10:
        return "Decagon"
    else:
        return "Polygon"

def detect_color(hsv, c):
    mask = np.zeros(hsv.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    h, s, v = cv2.mean(hsv, mask=mask)[:3]

    if v < 50:
        return "Black"
    elif s < 40 and v > 200:
        return "White"
    elif s < 40:
        return "Gray"
    elif h < 10 or h > 170:
        return "Red"
    elif 10 < h <= 25:
        return "Orange"
    elif 20 <= h <= 40:
        return "Yellow"
    elif 40 < h <= 85:
        return "Green"
    elif 85 < h <= 100:
        return "Cyan"
    elif 100 < h <= 130:
        return "Blue"
    elif 130 < h <= 160:
        return "Purple"
    elif 160 < h <= 170 and s > 150 and v > 150:
        return "Pink"
    else:
        return "Unknown"

def is_inside_focus(contour, frame_width, frame_height):
    fx, fy = frame_width // 2 - FOCUS_BOX_SIZE // 2, frame_height // 2 - FOCUS_BOX_SIZE // 2
    fw, fh = fx + FOCUS_BOX_SIZE, fy + FOCUS_BOX_SIZE
    x, y, w, h = cv2.boundingRect(contour)
    return fx <= x <= fw and fx <= x + w <= fw and fy <= y <= fh and fy <= y + h <= fh

def handle_mouse_click(event, x, y, flags, param):
    global quit_requested
    bx, by, bw, bh = QUIT_BUTTON_POS
    if event == cv2.EVENT_LBUTTONDOWN:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            quit_requested = True

def draw_diagonals(image, x, y, w, h):
    cv2.line(image, (x, y), (x + w, y + h), (255, 0, 255), 1)
    cv2.line(image, (x + w, y), (x, y + h), (255, 0, 255), 1)

def process_frame(frame):
    resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding and dilation for edge clarity
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = resized.shape[:2]
    fx, fy = frame_w // 2 - FOCUS_BOX_SIZE // 2, frame_h // 2 - FOCUS_BOX_SIZE // 2

    # Draw focus box and quit button
    cv2.rectangle(resized, (fx, fy), (fx + FOCUS_BOX_SIZE, fy + FOCUS_BOX_SIZE), (0, 255, 255), 2)
    cv2.putText(resized, "Focus Area", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    bx, by, bw, bh = QUIT_BUTTON_POS
    cv2.rectangle(resized, (bx, by), (bx + bw, by + bh), (0, 0, 255), -1)
    cv2.putText(resized, "QUIT", (bx + 10, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    detected = False
    detected_info = None

    for c in contours:
        if cv2.contourArea(c) < 150:
            continue
        if not is_inside_focus(c, frame_w, frame_h):
            continue

        shape = detect_shape(c)
        color = detect_color(hsv, c)
        x, y, w, h = cv2.boundingRect(c)
        w_cm = round(w / PIXELS_PER_CM, 1)
        h_cm = round(h / PIXELS_PER_CM, 1)
        dimensions = f"{w_cm}cm x {h_cm}cm"

        cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
        draw_diagonals(resized, x, y, w, h)
        cv2.putText(resized, f"{shape}, {color}, {dimensions}", (x, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)

        detected = True
        detected_info = (shape, color, w_cm, h_cm)
        break

    return resized, detected, detected_info

def save_screenshot(image):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(SAVE_PATH, f"detected_{timestamp}.png")
    cv2.imwrite(filename, image)
    print(f"[📸] Saved screenshot: {filename}")

def main():
    global quit_requested
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    cv2.namedWindow("Object Detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Object Detector", handle_mouse_click)

    cooldown = 3
    last_detection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, detected, info = process_frame(frame)
        cv2.imshow("Object Detector", output)

        current_time = time.time()
        if detected and current_time - last_detection_time > cooldown:
            shape, color, w_cm, h_cm = info
            announcement = f"Detected {shape}, {color}, size {w_cm} by {h_cm} centimeters"
            threading.Thread(target=speak, args=(announcement,), daemon=True).start()
            save_screenshot(output)
            last_detection_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q') or quit_requested:
            print("🛑 Quit requested. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()