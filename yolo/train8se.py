import sys
import argparse
import os

# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例
#yolov8_SEAttention.py
from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) # 直接加载yaml文件训练
    #model = YOLO(weights)  # 直接加载权重文件进行训练
    #model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练

    model.info()

    results = model.train(
    data='crop.yaml',
    epochs=100,
    imgsz=640,
    workers=4,
    batch=16,
    optimizer='AdamW',  # 👈 这里添加这一行
    lr0=0.001,          # 👈 建议设置较小的初始学习率
    weight_decay=0.0005
)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/v8/SEAtt_yolov8.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)