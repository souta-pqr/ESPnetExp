# ファイルを読み込む
with open('text', 'r') as f:
    lines = f.readlines()

# 各行から半角空白より前の部分を抽出し，アンダーバー二つ目までの部分を追加
extracted = [line.split(' ')[0] + " " + "_".join(line.split(' ')[0].split("_")[:2]) for line in lines]

# 結果をファイルに書き込む
with open('utt2spk', 'a') as f:
    for item in extracted:
        f.write("%s\n" % item)
