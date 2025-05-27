'''
Training for the helmet class using Yolov8
'''

from ultralytics import YOLO # to train the datasets by using YOLOv8
img_size = 640 # image size 
batch_size = 8 # batch size 
epochs = 50 # number of epochs, more is better but slower -_-

def main() -> None:
    model = YOLO('yolov8s.pt')
    results = model.train(
        data = 'datasets/helmet/data.yaml', # relative path
        imgsz = img_size,
        batch = batch_size,
        epochs = epochs,
        project = 'runs', 
        name= 'helmet' # saves to runs/helmet
    )
    print(f'Best Precision: {results.box.map*100:.2f}%')
    print('Completed!')

if __name__ == '__main__':
    main()
