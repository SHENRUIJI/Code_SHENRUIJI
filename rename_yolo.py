import os
from tqdm import tqdm

def rename_pair(img_dir, lbl_dir, prefix='data_', digits=4, start=1):
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        raise Exception("Invalid path.")

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    imgs = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts])
    labels = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

    if len(imgs) != len(labels):
        raise Exception(f"Image/label count mismatch: {len(imgs)} vs {len(labels)}")

    for i, (img, lbl) in enumerate(zip(imgs, labels), start=start):
        base = f"{prefix}{str(i).zfill(digits)}"
        img_new = base + os.path.splitext(img)[1]
        lbl_new = base + ".txt"

        os.rename(os.path.join(img_dir, img), os.path.join(img_dir, img_new))
        os.rename(os.path.join(lbl_dir, lbl), os.path.join(lbl_dir, lbl_new))

        tqdm.write(f"{img} -> {img_new}, {lbl} -> {lbl_new}")

    print(f"\nDone. Renamed {len(imgs)} pairs.")

if __name__ == "__main__":
    img_dir = r"D:\Projects\RICE_deaplearning\data\images"
    lbl_dir = r"D:\Projects\RICE_deaplearning\data\labels"
    rename_pair(img_dir, lbl_dir, prefix='crop_', digits=5, start=1)
