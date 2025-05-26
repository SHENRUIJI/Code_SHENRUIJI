import os

img_dir = r"D:\Projects\RICE_deaplearning\data\fuzuhe\Data Maize Growth Stage NEW\Pre_Grain"
prefix = "weed"

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
files.sort()

for i, fname in enumerate(files, 1):
    ext = os.path.splitext(fname)[1]
    new_name = f"{prefix}_{str(i).zfill(3)}{ext}"
    os.rename(os.path.join(img_dir, fname), os.path.join(img_dir, new_name))
    print(f"{fname} -> {new_name}")
