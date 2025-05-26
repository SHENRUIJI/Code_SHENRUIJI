import os

def update_yolo(yolo_dir, mapping):
    if not os.path.exists(yolo_dir):
        raise FileNotFoundError("找不到路径: " + yolo_dir)

    files = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.txt')])
    print("YOLO标签文件总数:", len(files))

    for i, name in enumerate(files, 1):
        path = os.path.join(yolo_dir, name)
        with open(path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid in mapping:
                parts[0] = str(mapping[cid])
            new_lines.append(' '.join(parts))

        with open(path, 'w') as f:
            f.write('\n'.join(new_lines))

        print(f"[{i}/{len(files)}] {name} 更新完成")

if __name__ == '__main__':
    yolo_dir = r"D:\Projects\RICE_deaplearning\数据集\处理中的数据集\水稻\txt"
    cls_map = {0: 8, 1: 9, 2: 10}
    update_yolo(yolo_dir, cls_map)
