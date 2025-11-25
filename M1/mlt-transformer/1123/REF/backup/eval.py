#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def combine_files(text_file_path, isdysfl_file_path, output_file_path):
    """
    textファイルとisdysflファイルを組み合わせて新しいファイルを作成
    
    Args:
        text_file_path (str): textファイルのパス
        isdysfl_file_path (str): isdysflファイルのパス
        output_file_path (str): 出力ファイルのパス
    """
    
    # タグマッピング
    tag_map = {
        '0': '',
        '1': '+F',
        '2': '+D',
        '3': '+I'
    }
    
    # ファイルを読み込んでIDでマッピング
    text_map = {}
    isdysfl_map = {}
    
    # textファイルを読み込み
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    id_part = parts[0]
                    words = parts[1:]
                    text_map[id_part] = words
    except FileNotFoundError:
        print(f"エラー: {text_file_path} が見つかりません")
        return
    except Exception as e:
        print(f"textファイル読み込みエラー: {e}")
        return
    
    # isdysflファイルを読み込み
    try:
        with open(isdysfl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    id_part = parts[0]
                    tags = parts[1:]
                    isdysfl_map[id_part] = tags
    except FileNotFoundError:
        print(f"エラー: {isdysfl_file_path} が見つかりません")
        return
    except Exception as e:
        print(f"isdysflファイル読み込みエラー: {e}")
        return
    
    # 結果を作成
    results = []
    
    for id_part in text_map:
        words = text_map[id_part]
        tags = isdysfl_map.get(id_part, [])
        
        if tags:
            # 各単語にタグを適用
            tagged_words = []
            for i, word in enumerate(words):
                if i < len(tags):
                    tag = tag_map.get(tags[i], '')
                    tagged_words.append(word + tag)
                else:
                    tagged_words.append(word)
            
            result_line = id_part + ' ' + ' '.join(tagged_words)
        else:
            # タグが見つからない場合はそのまま
            result_line = id_part + ' ' + ' '.join(words)
        
        results.append(result_line)
    
    # 結果をファイルに書き込み
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in results:
                f.write(line + '\n')
        
        print(f"処理完了: {output_file_path} に結果を保存しました")
        print(f"処理した行数: {len(results)}")
        
        # 結果の一部を表示
        print("\n--- 結果のプレビュー（最初の5行）---")
        for i, line in enumerate(results[:5]):
            print(line)
        if len(results) > 5:
            print("...")
            
    except Exception as e:
        print(f"出力ファイル書き込みエラー: {e}")

def main():
    """
    メイン関数 - ファイル名を指定して実行
    """
    # ファイル名を指定（必要に応じて変更してください）
    text_file = "text"
    isdysfl_file = "isdysfl"
    output_file = "combined_result"
    
    print("ファイル組み合わせツール")
    print("=" * 30)
    print(f"textファイル: {text_file}")
    print(f"isdysflファイル: {isdysfl_file}")
    print(f"出力ファイル: {output_file}")
    print()
    
    print("変換ルール:")
    print("0 → そのまま")
    print("1 → +F")
    print("2 → +D")
    print("3 → +I")
    print()
    
    # ファイル組み合わせ実行
    combine_files(text_file, isdysfl_file, output_file)

def interactive_mode():
    """
    対話モード - ファイル名を入力して実行
    """
    print("ファイル組み合わせツール（対話モード）")
    print("=" * 40)
    
    text_file = input("textファイル名を入力してください: ").strip()
    isdysfl_file = input("isdysflファイル名を入力してください: ").strip()
    output_file = input("出力ファイル名を入力してください（例: result.txt）: ").strip()
    
    if not output_file:
        output_file = "combined_result.txt"
    
    print()
    print("変換ルール:")
    print("0 → そのまま")
    print("1 → +F")
    print("2 → +D")
    print("3 → +I")
    print()
    
    combine_files(text_file, isdysfl_file, output_file)

# サンプルデータでテスト実行する関数
def test_with_sample_data():
    """
    サンプルデータでテスト実行
    """
    print("サンプルデータでテスト実行")
    print("=" * 30)
    
    # サンプルデータ作成
    sample_text = """K005_000_K005_001_IC03_0001831_0002202 ソ ー
K005_000_K005_001_IC03_0003240_0007756 エ ー ツ ク ッ タ ノ ワ ア ル ノ <sp> コ レ ワ ー <sp> ツ ク ッ タ ノ コ レ ワ ー <sp> ツ ク ッ タ ノ コ レ ワ
K005_000_K005_001_IC03_0012837_0013140 ウ ン ウ ン ウ ン ウ ン
K005_000_K005_001_IC03_0020266_0021809 コ レ ガ ニ ジュ ー ヨ ン ナ ノ コ ッ チ ワ
K005_000_K005_001_IC03_0023965_0025886 エ チュ ー チ ア ナ <sp> ア ー ノ ー
K005_000_K005_001_IC03_0029841_0030255 ウ ン
K005_000_K005_001_IC03_0032066_0032493 ウ ン
K005_000_K005_001_IC03_0037618_0038133 ウ ン"""

    sample_isdysfl = """K005_000_K005_001_IC03_0001831_0002202 0 0
K005_000_K005_001_IC03_0003240_0007756 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
K005_000_K005_001_IC03_0012837_0013140 3 3 3 3 3 3 3 3
K005_000_K005_001_IC03_0020266_0021809 0 0 0 0 0 0 0 0 0 0 0 0 0 0
K005_000_K005_001_IC03_0023965_0025886 0 0 0 1 0 0 1 1 1 1 0
K005_000_K005_001_IC03_0029841_0030255 3 3
K005_000_K005_001_IC03_0032066_0032493 3 3
K005_000_K005_001_IC03_0037618_0038133 3 3"""
    
    # サンプルファイル作成
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    with open("sample_isdysfl.txt", "w", encoding="utf-8") as f:
        f.write(sample_isdysfl)
    
    # テスト実行
    combine_files("sample_text.txt", "sample_isdysfl.txt", "sample_result.txt")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_mode()
        elif sys.argv[1] == "--test" or sys.argv[1] == "-t":
            test_with_sample_data()
        else:
            print("使用方法:")
            print("python script.py              # デフォルトファイル名で実行")
            print("python script.py --interactive # 対話モードで実行")
            print("python script.py --test       # サンプルデータでテスト")
    else:
        main()
