import os
import random
from PIL import Image, ImageFile
import torch
import numpy as np
from torchvision import transforms

def list_imgs(dir):
    return [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def load_labels(label_dir, img_file):
    path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return torch.empty((0, 5), dtype=torch.float32)
    with open(path, 'r', encoding='utf-8') as f:
        data = [list(map(float, line.strip().split())) for line in f if len(line.strip().split()) >= 5]
    return torch.tensor(data, dtype=torch.float32)

def save(img, boxes, out_dir, prefix, name):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    img.save(os.path.join(out_dir, 'images', prefix + name))
    with open(os.path.join(out_dir, 'labels', prefix + os.path.splitext(name)[0] + '.txt'), 'w', encoding='utf-8') as f:
        for cls, x, y, w, h in boxes.tolist():
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def add_count(counts, boxes):
    for b in boxes:
        cls = int(b[0])
        counts[cls] = counts.get(cls, 0) + 1

class Aug:
    def hflip(self, img, boxes):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes.size(0): boxes[:, 1] = 1.0 - boxes[:, 1]
        return img, boxes

    def vflip(self, img, boxes):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if boxes.size(0): boxes[:, 2] = 1.0 - boxes[:, 2]
        return img, boxes

    def crop(self, img, boxes, size=1024):
        w, h = img.size
        short = min(w, h)
        if not boxes.size(0):
            out = transforms.CenterCrop(short)(img)
            return out.resize((size, size)), boxes

        x = boxes[:, 1] * w
        y = boxes[:, 2] * h
        bw = boxes[:, 3] * w
        bh = boxes[:, 4] * h
        x1 = x - bw / 2
        y1 = y - bh / 2
        x2 = x + bw / 2
        y2 = y + bh / 2
        bxy = torch.stack([x1, y1, x2, y2], 1)
        delta = abs(w - h) / 2
        if w > h: bxy[:, [0, 2]] -= delta
        else: bxy[:, [1, 3]] -= delta
        bxy = bxy.clamp(0, short)
        keep = (bxy[:, 2] > bxy[:, 0]) & (bxy[:, 3] > bxy[:, 1])
        bxy = bxy[keep]
        cls = boxes[keep, 0] if keep.any() else torch.empty(0)
        img = transforms.CenterCrop(short)(img).resize((size, size))
        bxy *= size / short
        new = []
        for i, box in enumerate(bxy):
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2 / size
            yc = (y1 + y2) / 2 / size
            bw = (x2 - x1) / size
            bh = (y2 - y1) / size
            new.append([cls[i].item(), xc, yc, bw, bh])
        return img, torch.tensor(new, dtype=torch.float32)

    def tensor_aug(self, img):
        t = transforms.ToTensor()(img)
        funcs = [
            lambda x: (x + np.random.uniform(-40, 40)/255).clamp(0, 1),
            lambda x: (x * np.random.uniform(0.5, 1.5)).clamp(0, 1),
            lambda x: x + torch.normal(0, 0.1, x.shape),
            lambda x: torch.where(torch.rand_like(x) > 0.9, torch.ones_like(x), x),
            lambda x: torch.where(torch.rand_like(x) > 0.9, torch.zeros_like(x), x)
        ]
        return [(transforms.ToPILImage()(f(t)),) for f in funcs]

def scan_labels(label_dir):
    count, cls_img = {}, {}
    for f in os.listdir(label_dir):
        if not f.endswith('.txt'): continue
        with open(os.path.join(label_dir, f), 'r', encoding='utf-8') as fi:
            for line in fi:
                p = line.strip().split()
                if len(p) < 5: continue
                c = int(float(p[0]))
                count[c] = count.get(c, 0) + 1
                cls_img.setdefault(c, set()).add(os.path.splitext(f)[0])
    return count, cls_img

def run(img_dir, label_dir, out_dir, target=10000, seed=42):
    random.seed(seed)
    counts, cls_imgs = scan_labels(label_dir)
    if not counts:
        print("无标签")
        return

    print("初始:", counts)
    aug = Aug()
    for cls in sorted(counts):
        print(f"增强类别 {cls} ...")
        while counts[cls] < target:
            names = list(cls_imgs.get(cls, []))
            if not names: break
            name = random.choice(names)
            imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[0] == name]
            if not imgs: continue
            img_path = os.path.join(img_dir, imgs[0])
            lbl_path = os.path.join(label_dir, name + '.txt')
            if not os.path.exists(img_path) or not os.path.exists(lbl_path): continue
            img = Image.open(img_path)
            boxes = load_labels(label_dir, imgs[0])
            out_data = {
                'hf': aug.hflip(img.copy(), boxes.clone()),
                'vf': aug.vflip(img.copy(), boxes.clone()),
                'cp': aug.crop(img.copy(), boxes.clone(), 1024),
            }
            for i, (im,) in enumerate(aug.tensor_aug(img)):
                out_data[f't{i}'] = (im, boxes.clone())
            for tag, (im, bx) in out_data.items():
                prefix = f"aug{cls}_{random.randint(0,9999)}_{tag}_"
                save(im, bx, out_dir, prefix, imgs[0])
                add_count(counts, bx)
            if counts[cls] >= target:
                print(f"类别 {cls} 完成，共 {counts[cls]} 条")
                break

    print("完成:")
    for k in sorted(counts):
        print(f"类别 {k}: {counts[k]}")

if __name__ == "__main__":
    imgs = r"D:\\Projects\\RICE_deaplearning\\datasets\\处理中的数据集\\杂草\\image"
    labels = r"D:\\Projects\\RICE_deaplearning\\datasets\\处理中的数据集\\杂草\\yolo_label"
    out = r"D:\\Projects\\RICE_deaplearning\\data"
    run(imgs, labels, out, target=10000)
