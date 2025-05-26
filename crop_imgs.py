import os
import cv2

img_dir = r"D:\Projects\RICE_deaplearning\data\yolo\val\images"
label_dir = r"D:\Projects\RICE_deaplearning\data\yolo\val\labels"
out_dir = r"D:\Projects\RICE_deaplearning\data\resnetClass\test"

# 遍历
for name in os.listdir(img_dir):
    if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, name)
    label_path = os.path.join(label_dir, os.path.splitext(name)[0] + '.txt')

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图像：", img_path)
        continue

    H, W = img.shape[:2]

    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = parts
            cls = int(cls)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            # 转换成像素坐标
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            cls_folder = os.path.join(out_dir, f"class_{cls}")
            os.makedirs(cls_folder, exist_ok=True)

            out_name = f"{os.path.splitext(name)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(cls_folder, out_name), crop)

print("处理完成，分类用数据已生成。")
