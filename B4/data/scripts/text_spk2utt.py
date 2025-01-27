def process_text(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.read().split('\n')  # テキストを行ごとに分割

    result = []  # 結果を保存するためのリスト
    current_prefix = None  # 現在のプレフィックスを保存するための変数
    current_line = ""  # 現在の行を保存するための変数

    for line in lines:
        prefix = "_".join(line.split('_')[:2])  # プレフィックスを取得（アンダーバーまで）
        if prefix != current_prefix:  # 新しいプレフィックスが見つかった場合
            if current_prefix is not None:  # 現在のプレフィックスがNoneでない場合、結果を追加
                result.append(current_prefix + " " + current_line)
            current_prefix = prefix  # 現在のプレフィックスを更新
            current_line = line.split(' ')[0]  # 現在の行を更新
        else:
            current_line += " " + line.split(' ')[0]  # 現在の行に新しい行を追加

    if current_line:  # 最後のプレフィックスの行を追加
        result.append(current_prefix + " " + current_line)

    with open(output_file, 'w') as f:
        f.write('\n'.join(result))  # 結果をファイルに書き込む

# ファイルを処理
process_text('text', 'spk2utt')
