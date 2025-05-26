import os
import random
import shutil

# 原始图像和标签目录
img_dir = r"D:\Projects\RICE_deaplearning\data\fuzuhe\images"
lbl_dir = r"D:\Projects\RICE_deaplearning\data\fuzuhe\labels"

# 拆分后的目标目录
t_img = r"D:\Projects\RICE_deaplearning\data\fuzuhe\fuzuhedata\train\images"
t_lbl = r"D:\Projects\RICE_deaplearning\data\fuzuhe\fuzuhedata\train\labels"
v_img = r"D:\Projects\RICE_deaplearning\data\fuzuhe\fuzuhedata\val\images"
v_lbl = r"D:\Projects\RICE_deaplearning\data\fuzuhe\fuzuhedata\val\labels"

n_cls = 11
t_ratio = 0.9
copy_files = True
seed = 2025

def main():
    random.seed(seed)

    files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    files.sort()

    def to_img(txt):
        return os.path.splitext(txt)[0] + '.jpg'

    data = []
    total = [0] * n_cls

    for f in files:
        l_path = os.path.join(lbl_dir, f)
        i_name = to_img(f)
        i_path = os.path.join(img_dir, i_name)

        if not os.path.exists(i_path):
            print("跳过缺图:", i_path)
            continue

        cnt = [0] * n_cls
        with open(l_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                cls = int(parts[0])
                cnt[cls] += 1

        for i in range(n_cls):
            total[i] += cnt[i]

        data.append({'img': i_name, 'lbl': f, 'cnt': cnt})

    tgt_t = [round(x * t_ratio) for x in total]
    tgt_v = [total[i] - tgt_t[i] for i in range(n_cls)]

    used_t = [0] * n_cls
    used_v = [0] * n_cls

    random.shuffle(data)
    set_t = []
    set_v = []

    for d in data:
        sc_t = 0
        sc_v = 0
        for i in range(n_cls):
            nt = tgt_t[i] - used_t[i]
            nv = tgt_v[i] - used_v[i]
            if nt > 0:
                sc_t += min(nt, d['cnt'][i])
            if nv > 0:
                sc_v += min(nv, d['cnt'][i])

        if sc_t >= sc_v:
            set_t.append(d)
            for i in range(n_cls):
                used_t[i] += d['cnt'][i]
        else:
            set_v.append(d)
            for i in range(n_cls):
                used_v[i] += d['cnt'][i]

    def sum_cnt(data):
        res = [0] * n_cls
        for d in data:
            for i in range(n_cls):
                res[i] += d['cnt'][i]
        return res

    cnt_t = sum_cnt(set_t)
    cnt_v = sum_cnt(set_v)

    print("\n最终统计：")
    for i in range(n_cls):
        tot = total[i]
        if tot == 0:
            continue
        print(f"类 {i}: 总 {tot}, 训练 {cnt_t[i]} ({cnt_t[i]/tot*100:.1f}%), 验证 {cnt_v[i]} ({cnt_v[i]/tot*100:.1f}%)")

    print(f"\n训练集: {len(set_t)} 张, 验证集: {len(set_v)} 张")

    if copy_files:
        os.makedirs(t_img, exist_ok=True)
        os.makedirs(t_lbl, exist_ok=True)
        os.makedirs(v_img, exist_ok=True)
        os.makedirs(v_lbl, exist_ok=True)

        def copy(data, out_img, out_lbl):
            for d in data:
                shutil.copy2(os.path.join(img_dir, d['img']), os.path.join(out_img, d['img']))
                shutil.copy2(os.path.join(lbl_dir, d['lbl']), os.path.join(out_lbl, d['lbl']))

        print("\n复制训练集...")
        copy(set_t, t_img, t_lbl)
        print("复制验证集...")
        copy(set_v, v_img, v_lbl)
        print("完成。")

if __name__ == '__main__':
    main()
