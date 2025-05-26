import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml)
    
    model.info()
    
    results = model.train(
    data='crop.yaml',
    epochs=100,
    imgsz=640,
    workers=4,
    batch=16,
    optimizer='AdamW', 
    lr0=0.001,     
    weight_decay=0.0005
)

'''
    results = model.train(data='crop.yaml', 
                        epochs=100, 
                        imgsz=640, 
                        workers=4, 
                        batch=16,
                        )
'''
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/11/yolo11n.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
