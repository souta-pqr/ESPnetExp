def check_d_tags(input_filename, output_filename):
    # 入力ファイルを読み込み、出力ファイルを書き込みモードで開く
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        # 各行をチェック
        for line in infile:
            # 開始タグと終了タグの数を数える
            start_tags = line.count('(F)')
            end_tags = line.count('(/F)')
            
            # タグの数が不一致の場合、その行を出力ファイルに書き込む
            if start_tags != end_tags:
                outfile.write(line)

# メイン処理
if __name__ == "__main__":
    input_file = "text"
    output_file = "check.txt"
    
    try:
        check_d_tags(input_file, output_file)
        print(f"処理が完了しました。結果は{output_file}に保存されています。")
    except FileNotFoundError:
        print("入力ファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")