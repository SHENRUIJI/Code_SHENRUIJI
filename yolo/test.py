from ultralytics import YOLO

# 加载模型
model = YOLO("/root/ultralytics-8.3.27/runs/detect/train4/weights/best.pt")

# 运行推理并仅保留crop类别
results = model(
    ["/root/ultralytics-8.3.27/southeast.jpg"],
    classes=[0], 
    conf=0.5    
)

# 处理结果
for result in results:
    result.show()
    result.save(filename="/root/ultralytics-8.3.27/test/result.jpg")