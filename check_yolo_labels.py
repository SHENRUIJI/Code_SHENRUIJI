import os

img_dir = r"D:\Projects\RICE_deaplearning\data\images"
lbl_dir = r"D:\Projects\RICE_deaplearning\data\labels"
img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts]
lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

img_set = {os.path.splitext(f)[0] for f in img_files}
lbl_set = {os.path.splitext(f)[0] for f in lbl_files}

missing_lbl = img_set - lbl_set
orphan_lbl = lbl_set - img_set
empty_lbl = []

for f in lbl_files:
    with open(os.path.join(lbl_dir, f), 'r') as fh:
        if not fh.read().strip():
            empty_lbl.append(f)

print(f"Total images: {len(img_set)}")
print(f"Total labels: {len(lbl_set)}\n")

if missing_lbl:
    print("Images with no label file:")
    for name in sorted(missing_lbl):
        print(f"  {name}")

if orphan_lbl:
    print("\nLabel files with no image:")
    for name in sorted(orphan_lbl):
        print(f"  {name}")

if empty_lbl:
    print("\nEmpty label files:")
    for f in sorted(empty_lbl):
        print(f"  {f}")
