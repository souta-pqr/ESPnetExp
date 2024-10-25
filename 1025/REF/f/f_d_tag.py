import re

def process_line(line):
    # +F と +D のグループを見つけるパターン
    pattern = r'(\S+(?:\+[F])+)'
    
    def replace_tags(match):
        word = match.group(1)
        base = word.rstrip('F+')
        tags = re.findall(r'\+([F])', word)
        
        result = base
        for tag in tags:
            result = f'<{tag}> {result} </{tag}>'
        
        return result
    
    # パターンを置換
    processed_line = re.sub(pattern, replace_tags, line)
    
    return processed_line.strip()

# ファイルを読み込んで処理し、結果を新しいファイルに書き込む
with open('text', 'r', encoding='utf-8') as input_file, open('changed_text', 'w', encoding='utf-8') as output_file:
    for line in input_file:
        processed_line = process_line(line)
        output_file.write(processed_line + '\n')

print("処理が完了しました。結果は'changed_text'ファイルに保存されています。")