import os
from collections import defaultdict

label_dir = "/root/autodl-tmp/newyolodata/train/labels"
class_stats = defaultdict(int)  
file_stats = defaultdict(int) 
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        path = os.path.join(label_dir, label_file)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # 记录原始行数和类别
        original_count = len(lines)
        line_counter = defaultdict(int)
        for line in lines:
            if line.strip(): 
                cls_id = line.split()[0]  
                line_counter[line] += 1
        
        unique_lines = []
        removed = 0
        for line, count in line_counter.items():
            unique_lines.append(line)
            if count > 1:
                cls_id = line.split()[0]
                class_stats[cls_id] += (count - 1)
                removed += (count - 1)
        
        if removed > 0:
            file_stats[label_file] = removed
        

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
