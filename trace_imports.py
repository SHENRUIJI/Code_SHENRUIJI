import sys
import os

called_files = set()

def trace_func(frame, event, arg):
    if event == "call":
        filename = frame.f_globals.get("__file__")
        if filename and filename.endswith(".py") and "site-packages" not in filename:
            abs_path = os.path.abspath(filename)
            called_files.add(abs_path)
    return trace_func

def print_result():
    output_file = "trace_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("--- Called Python Files ---\n")
        for file in sorted(called_files):
            f.write(f"{file}\n")
    print(f"Saved trace results to {output_file}")

import atexit
atexit.register(print_result)

sys.settrace(trace_func)

# 替换为你项目的入口文件（不要加 .py）
import Resnet
