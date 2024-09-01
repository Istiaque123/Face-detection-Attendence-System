from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    # use the root path for yaml
    model.train(data = "Dataset/SplitData/dataOffline_v2.yaml", epochs = 3)


if __name__ == '__main__':
    # print("Check")
    main()
    # print("Success")
