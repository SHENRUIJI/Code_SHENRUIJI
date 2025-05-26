import os
from collections import defaultdict

label_dir = "/root/autodl-tmp/newyolodata/train/labels"
class_stats = defaultdict(int)  # 统计每个类别被删除的数量
file_stats = defaultdict(int)   # 统计每个文件删除的数量

for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        path = os.path.join(label_dir, label_file)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # 记录原始行数和类别
        original_count = len(lines)
        line_counter = defaultdict(int)
        for line in lines:
            if line.strip():  # 忽略空行
                cls_id = line.split()[0]  # 提取类别ID
                line_counter[line] += 1
        
        # 去重处理
        unique_lines = []
        removed = 0
        for line, count in line_counter.items():
            unique_lines.append(line)
            if count > 1:
                # 统计被删除的重复项
                cls_id = line.split()[0]
                class_stats[cls_id] += (count - 1)  # 记录该类别被删除的数量
                removed += (count - 1)
        
        # 更新文件删除统计
        if removed > 0:
            file_stats[label_file] = removed
        
        # 写入去重后的内容
        with open(path, 'w') as f:
            f.writelines(unique_lines)

# 输出统计结果
print("\nDeleted duplicate labels statistics:")
print("------------------------------------")
for cls_id, count in class_stats.items():
    print(f"Class {cls_id}: {count} duplicates removed")

print("\nFiles with duplicates removed:")
print("------------------------------")
for filename, count in file_stats.items():
    print(f"{filename}: {count} duplicates")