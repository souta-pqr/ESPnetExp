import re

def merge_consecutive_tags(line):
    line = re.sub(r'</D>\s*<D>', ' ', line)
    return line

# ファイルを読み込んで処理し、結果を新しいファイルに書き込む
with open('text', 'r', encoding='utf-8') as input_file, open('changed_text', 'w', encoding='utf-8') as output_file:
    for line in input_file:
        processed_line = merge_consecutive_tags(line.strip())
        output_file.write(processed_line + '\n')

print("処理が完了しました。結果は'changed_text'ファイルに保存されています。")