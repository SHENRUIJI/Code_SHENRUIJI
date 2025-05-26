import os
import json
from PIL import Image

dst_dir = r"D:\Projects\Pycharm_projects\nil\database\coco_crop"
src_dir = r"D:\Projects\Pycharm_projects\nil\database\data_yolov8"
img_root = os.path.join(src_dir, "images")
lbl_root = os.path.join(src_dir, "labels")

# 类别表
categories = [
    {"id": 0, "name": "crop"},
    {"id": 1, "name": "weed1"},
    {"id": 2, "name": "weed2"}
]

def yolo2coco(x, y, w, h, img_w, img_h):
    return [
        (x - w / 2) * img_w,
        (y - h / 2) * img_h,
        w * img_w,
        h * img_h
    ]

# main
for split in ['train', 'val']:
    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1

    split_img_dir = os.path.join(img_root, split)
    if not os.path.exists(split_img_dir):
        print(f"Skip: {split_img_dir} not found.")
        continue

    for fname in os.listdir(split_img_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(split_img_dir, fname)
        lbl_path = os.path.join(lbl_root, split, os.path.splitext(fname)[0] + ".txt")

        with Image.open(img_path) as im:
            w, h = im.size

        img_id = len(coco["images"]) + 1
        coco["images"].append({
            "file_name": fname,
            "id": img_id,
            "width": w,
            "height": h
        })

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                box = yolo2coco(xc, yc, bw, bh, w, h)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                })
                ann_id += 1

    out_path = os.path.join(dst_dir, f"{split}_coco_format.json")
    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"[OK] {split} annotations saved: {out_path}")
