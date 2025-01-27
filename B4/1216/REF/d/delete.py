import re

# changed_textファイルを読み込む
with open('text', 'r') as file:
    changed_text = file.read()

# 連続する2つの半角空白を1つの半角空白に置換する
modified_text = re.sub(r'  ', ' ', changed_text)

# 修正後のテキストを新しいファイルに書き込む
with open('changed_text', 'w') as file:
    file.write(modified_text)