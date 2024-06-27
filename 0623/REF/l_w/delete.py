import re

def process_line(line):
    # +Wのパターン
    plus_w_pattern = r'\+D'
    
    # +Wを削除
    processed_line = re.sub(plus_w_pattern, '', line)
    # 連続する空白を1つにまとめる
    processed_line = re.sub(r'\s+', ' ', processed_line)
    
    return processed_line.strip()

# ファイルを読み込んで処理し、結果を新しいファイルに書き込む
with open('text', 'r', encoding='utf-8') as input_file, open('changed_text', 'w', encoding='utf-8') as output_file:
    for line in input_file:
        processed_line = process_line(line)
        output_file.write(processed_line + '\n')

print("処理が完了しました。結果は'changed_text'ファイルに保存されています。")