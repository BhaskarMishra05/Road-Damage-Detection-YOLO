from ultralytics import YOLO
import os
model = YOLO('yolo26n.pt')
path_yml = os.path.join('data','data.yaml')

model.train(
    data = path_yml,
    device = 0
)