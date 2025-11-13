from ultralytics import YOLO

def main():
    model = YOLO("yolov5n.pt")

    data_path = r'D:\BCTT\yolov5\data_CanTho_v5\data.yaml'
    
    try:
        results = model.train(data=data_path, epochs=10, imgsz=640, batch=16, pretrained=True, val=True, device='cpu')

        print("Training completed.")
    except Exception as e:
        print("Co loi xay ra:", str(e))

if __name__ == '__main__':
    main()
