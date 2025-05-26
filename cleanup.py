import os

img_dir = r"D:\Projects\RICE_deaplearning\data\yolo\train\images"
lbl_dir = r"D:\Projects\RICE_deaplearning\data\yolo\train\labels"

# 拿文件名
imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))}
lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

extra_imgs = imgs - lbls
extra_lbls = lbls - imgs

# 清理图片
for name in extra_imgs:
    for ext in ('.jpg', '.png'):
        path = os.path.join(img_dir, name + ext)
        if os.path.exists(path):
            print(f"Removing image: {path}")
            os.remove(path)
            break

for name in extra_lbls:
    path = os.path.join(lbl_dir, name + '.txt')
    if os.path.exists(path):
        print(f"Removing label: {path}")
        os.remove(path)
