def convert_text_with_disfluency(text_lines, disfluency_lines):
    """
    textとdisfluencyのデータを結合して、convert-textを作成する関数
    
    Args:
        text_lines (list): textファイルの各行のリスト
        disfluency_lines (list): disfluencyファイルの各行のリスト
    
    Returns:
        list: 変換されたテキストの各行のリスト
    """
    
    disfluency_map = {
        '1': '+F',
        '2': '+D',  
        '3': '+I',
        '0': ''  # 0の場合は何も付加しない
    }
    
    converted_lines = []
    
    for text_line, disfluency_line in zip(text_lines, disfluency_lines):
        # 行をスペースで分割
        text_parts = text_line.strip().split()
        disfluency_parts = disfluency_line.strip().split()
        
        # IDを取得（最初の部分）
        line_id = text_parts[0]
        
        # テキスト部分（IDを除く）
        text_tokens = text_parts[1:]
        
        # disfluency部分（IDを除く）
        disfluency_codes = disfluency_parts[1:]
        
        # テキストトークンとdisfluencyコードの数が一致するかチェック
        if len(text_tokens) != len(disfluency_codes):
            print(f"Warning: Length mismatch in line {line_id}")
            print(f"Text tokens: {len(text_tokens)}, Disfluency codes: {len(disfluency_codes)}")
            continue
        
        # 変換されたテキストを構築
        converted_tokens = []
        for token, code in zip(text_tokens, disfluency_codes):
            if code in disfluency_map and disfluency_map[code]:
                converted_tokens.append(token + disfluency_map[code])
            else:
                converted_tokens.append(token)
        
        # 行を再構築
        converted_line = line_id + ' ' + ' '.join(converted_tokens)
        converted_lines.append(converted_line)
    
    return converted_lines

# サンプルデータでテスト
def test_converter():
    # サンプルのtextデータ
    text_data = [
        "K005_000_K005_001_IC03_0001831_0002202 ソ ー",
        "K005_000_K005_001_IC03_0003240_0007756 エ ー ツ ク ッ タ ノ ワ ア ル ノ <sp> コ レ ワ ー <sp> ツ ク ッ タ ノ コ レ ワ ー <sp> ツ ク ッ タ ノ コ レ ワ",
        "K005_000_K005_001_IC03_0012837_0013140 ウ ン ウ ン ウ ン ウ ン",
        "K005_000_K005_001_IC03_0049393_0051832 え ー そ そ う で す ね"
    ]
    
    # サンプルのdisfluencyデータ
    disfluency_data = [
        "K005_000_K005_001_IC03_0001831_0002202 0 0",
        "K005_000_K005_001_IC03_0003240_0007756 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "K005_000_K005_001_IC03_0012837_0013140 3 3 3 3 3 3 3 3",
        "K005_000_K005_001_IC03_0049393_0051832 1 1 2 0 0 0 0 0"
    ]
    
    # 変換実行
    result = convert_text_with_disfluency(text_data, disfluency_data)
    
    print("変換結果:")
    for line in result:
        print(line)

# ファイルから読み込んで処理する関数
def process_files(text_file_path, disfluency_file_path, output_file_path):
    """
    ファイルから読み込んで処理し、結果を出力ファイルに保存
    """
    try:
        # ファイルを読み込み
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
        
        with open(disfluency_file_path, 'r', encoding='utf-8') as f:
            disfluency_lines = f.readlines()
        
        # 変換実行
        converted_lines = convert_text_with_disfluency(text_lines, disfluency_lines)
        
        # 結果を出力ファイルに保存
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in converted_lines:
                f.write(line + '\n')
        
        print(f"変換完了: {output_file_path}")
        
    except Exception as e:
        print(f"エラー: {e}")

# テスト実行
if __name__ == "__main__":
    test_converter()
    
    # ファイル処理の使用例
    process_files('text', 'disfluency', 'convert-text')
