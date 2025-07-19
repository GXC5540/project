file_path = r"C:\Users\HUAWEI\Desktop\cnews.train4.txt"

# 读取并处理文件
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
removed_count = 0

for line in lines:
    parts = line.strip().split('\t')
    if len(parts) >= 2:
        label, content = parts[0], parts[1]
        # 如果不是家居类，或家居类但字数≥50，则保留
        if label != '家居' or (label == '家居' and len(content) >= 50):
            new_lines.append(line)
        else:
            removed_count += 1

# 覆盖写入原文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)  # 保留原有换行格式

# 打印结果
print(f"已删除 {removed_count} 条家居类且字数<50的短文本")
print(f"剩余 {len(new_lines)} 条有效数据")
print(f"文件已更新：{file_path}")