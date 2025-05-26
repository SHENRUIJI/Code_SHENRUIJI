import os
import cv2
from collections import defaultdict

def count_cls(lbl_dir):
    if not os.path.exists(lbl_dir):
        raise FileNotFoundError("路径不存在: " + lbl_dir)

    files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    print("总文件数:", len(files))

    stats = defaultdict(int)
    for i, name in enumerate(files, 1):
        path = os.path.join(lbl_dir, name)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                stats[cls] += 1
        print(f"[{i}/{len(files)}] 处理完: {name}")
    return stats

def compress_imgs(in_dir, out_dir, max_kb=150, max_w=800):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(in_dir):
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        in_path = os.path.join(in_dir, name)
        out_path = os.path.join(out_dir, name)

        img = cv2.imread(in_path)
        if img is None:
            print("跳过无法读取的图像:", name)
            continue

        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        q = 95
        while q >= 20:
            cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if os.path.getsize(out_path) / 1024 <= max_kb:
                break
            q -= 5

        print(f"{name} -> {os.path.getsize(out_path) // 1024}KB，质量: {q}")

if __name__ == '__main__':
    lbl_dir = r"D:\\Projects\\RICE_deaplearning\\data\\yolo\\train\\labels"
    result = count_cls(lbl_dir)
    print("\n类别统计:")
    for cls, cnt in result.items():
        print(f"类 {cls}: {cnt} 个")

    in_dir = r"D:\\Projects\\RICE_deaplearning\\data\\处理中的数据\images"
    out_dir = r"D:\\Projects\\RICE_deaplearning\\data\images"
    compress_imgs(in_dir, out_dir)
