from ultralytics import YOLO

input_path = r"D:\BCTT\yolov5\video_demo\NKKN-VoThiSau 2017-07-18_08_00_00_000.asf"

model = YOLO(r"D:\BCTT\yolov5\runs\detect\train_customdata\weights\best.pt")
results = model.predict(input_path, conf=0.4, save=True, show=True, imgsz=800)
