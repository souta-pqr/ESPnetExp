def process_text_file(filename):
    """
    指定された条件で行を処理する関数
    条件：(R, (F, (I, (Dが含まれていない行で、)がある場合その行を出力
    """
    try:
        # ファイルを読み込む
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 除外すべきパターンを定義
        exclude_patterns = ['(R', '(F', '(I', '(D']
        
        # 各行を処理
        for line in lines:
            # 改行文字を削除
            line = line.rstrip('\n')
            
            # 空行をスキップ
            if not line.strip():
                continue
            
            # 条件チェック
            # 1. 除外パターンが含まれていないことを確認
            has_exclude_pattern = any(pattern in line for pattern in exclude_patterns)
            
            # 2. )が含まれていることを確認
            has_closing_paren = ')' in line
            
            # 両方の条件を満たす場合、行を出力
            if not has_exclude_pattern and has_closing_paren:
                print(line)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# メインの実行部分
process_text_file('text')