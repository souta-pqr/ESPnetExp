import re

def count_tags(text):
    """
    テキスト内の (I 文字), (F 文字), (D 文字) の形式のタグを数える関数
    
    Args:
        text (str): 入力テキスト
    
    Returns:
        dict: 各タグの出現回数
    """
    # タグのパターンを定義
    i_pattern = r'\(I\s[^\)]+\)'
    f_pattern = r'\(F\s[^\)]+\)'
    d_pattern = r'\(D\s[^\)]+\)'
    
    # 各パターンの出現回数を数える
    i_count = len(re.findall(i_pattern, text))
    f_count = len(re.findall(f_pattern, text))
    d_count = len(re.findall(d_pattern, text))
    
    return {
        'I_tags': i_count,
        'F_tags': f_count,
        'D_tags': d_count
    }

def main():
    # 入力ファイルを読み込む
    try:
        with open('text', 'r', encoding='utf-8') as f:
            input_text = f.read()
    except FileNotFoundError:
        print("Error: 入力ファイル 'text' が見つかりません。")
        return
    except Exception as e:
        print(f"Error: ファイル読み込み中にエラーが発生しました: {e}")
        return

    # タグを数える
    result = count_tags(input_text)
    
    # 結果をファイルに書き出す
    try:
        with open('output.txt', 'w', encoding='utf-8') as f:
            for tag, count in result.items():
                f.write(f"{tag}: {count}\n")
        print("結果を output.txt に書き出しました。")
    except Exception as e:
        print(f"Error: 結果の書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()