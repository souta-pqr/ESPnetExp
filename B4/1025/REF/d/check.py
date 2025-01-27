def compare_text_files(file1_path, file2_path, output_path):
    """
    file1_pathにあってfile2_pathにない行、または異なる行を抽出してoutput_pathに出力する
    
    Args:
        file1_path (str): 比較元のファイルパス
        file2_path (str): 比較対象のファイルパス 
        output_path (str): 出力ファイルパス
    """
    # ファイルの内容を読み込む
    with open(file1_path, 'r', encoding='utf-8') as f1:
        text1_lines = f1.readlines()
    
    with open(file2_path, 'r', encoding='utf-8') as f2:
        text2_lines = f2.readlines()
    
    # text2の行を辞書に格納（高速な検索のため）
    text2_dict = {line.strip(): True for line in text2_lines}
    
    # 差分を格納するリスト
    differences = []
    
    # text1の各行について、text2に存在するか確認
    for line in text1_lines:
        line = line.strip()
        if line not in text2_dict:
            differences.append(line)
    
    # 結果をファイルに書き出し
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in differences:
            f_out.write(line + '\n')
        
# 使用例
if __name__ == "__main__":
    compare_text_files('check/text', 'text', 'output.txt')