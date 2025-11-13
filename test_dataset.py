import cv2
from ultralytics import YOLO

# Load 2 model
model_trained = YOLO(r"D:\yolov5\runs\detect\train_customdata\weights\best.pt")  # Model đã fine-tune
model_pretrained = YOLO(r"yolov5s.pt")  # Model gốc chưa fine-tune

# Mở video
cap = cv2.VideoCapture(r"D:\yolov5\video_demo\NKKN-VoThiSau 2017-07-18_08_00_00_000.asf")

target_classes = [1, 2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán với cả hai mô hình
    results_trained = model_trained.predict(frame, conf=0.25, verbose=False)
    results_pretrained = model_pretrained.predict(frame, conf=0.25, verbose=False, classes=target_classes)

    # Vẽ kết quả
    frame_trained = results_trained[0].plot()
    frame_pretrained = results_pretrained[0].plot()

    # Gộp hai kết quả song song
    combined = cv2.hconcat([frame_pretrained, frame_trained])
    cv2.imshow("Left: Pretrained | Right: Fine-tuned", combined)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
