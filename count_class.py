import os
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

if __name__ == '__main__':
    lbl_dir = r"D:\\Projects\\RICE_deaplearning\\data\\yolo\\train\\labels"
    result = count_cls(lbl_dir)
    print("\n类别统计:")
    for cls, cnt in result.items():
        print(f"类 {cls}: {cnt} 个")
