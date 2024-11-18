import re

# 笑いのパターン
laughter_pattern = r'\(F\)\s*([^(/]+)\s*\(/F\)'

# ファイルを読み込む
with open('text', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 各行から笑いのパターンを削除
changed_lines = []
for line in lines:
    # 笑いのパターンを削除し、内容のみを残す
    changed_line = re.sub(laughter_pattern, r'\1', line)
    changed_lines.append(changed_line)

# 変更された内容を新しいファイルに書き込む
with open('changed_text', 'w', encoding='utf-8') as file:
    file.writelines(changed_lines)