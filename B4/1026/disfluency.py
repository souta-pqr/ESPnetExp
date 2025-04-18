import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()

    # Dとカッコの間の文字列を抽出する正規表現パターン
    d_pattern = r'\(D\s+([^)]+)\)'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("d_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにDの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)
            sentences2 = re.split(sentence_pattern, text2)

            total_d_ref_count = 0
            total_d_hyp_count = 0
            total_d_correct_count = 0
            total_d_correct_count_relaxed = 0
            total_d_deletion_count = 0
            total_d_insertion_count = 0
            total_d_deletion_count_relaxed = 0
            total_d_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]
                sentence2 = sentences2[i+1]

                # 正解文と音声認識文からDの単語を抽出
                d_ref_words = re.findall(d_pattern, sentence1)
                d_hyp_words = re.findall(d_pattern, sentence2)

                # DPを使ってアライメントを取得
                d_alignment = align_words(d_ref_words, d_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                d_ref_count = len(d_ref_words)
                d_hyp_count = len(d_hyp_words)
                d_correct_count = sum(1 for ref, hyp in d_alignment if ref == hyp)
                d_correct_count_relaxed = sum(1 for ref, hyp in d_alignment if ref and hyp)
                d_deletion_count = d_ref_count - d_correct_count
                d_insertion_count = d_hyp_count - d_correct_count
                d_deletion_count_relaxed = d_ref_count - d_correct_count_relaxed
                d_insertion_count_relaxed = d_hyp_count - d_correct_count_relaxed

                # Precision、Recall、F1スコアを計算
                d_precision = d_correct_count / d_hyp_count * 100 if d_hyp_count > 0 else 0
                d_recall = d_correct_count / d_ref_count * 100 if d_ref_count > 0 else 0
                d_precision_relaxed = d_correct_count_relaxed / d_hyp_count * 100 if d_hyp_count > 0 else 0
                d_recall_relaxed = d_correct_count_relaxed / d_ref_count * 100 if d_ref_count > 0 else 0

                # F1スコアを計算
                d_f1 = 2 * d_precision * d_recall / (d_precision + d_recall) if d_precision + d_recall > 0 else 0
                d_f1_relaxed = 2 * d_precision_relaxed * d_recall_relaxed / (d_precision_relaxed + d_recall_relaxed) if d_precision_relaxed + d_recall_relaxed > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {sentence1}")
                out_file.write(f"HYP: {sentence2}")
                out_file.write(f"言い直しの検出率 - Precision: {d_precision:.2f}%, Recall: {d_recall:.2f}%, F1: {d_f1:.2f}%\n")
                out_file.write(f"言い直しの検出率 (緩和) - Precision: {d_precision_relaxed:.2f}%, Recall: {d_recall_relaxed:.2f}%, F1: {d_f1_relaxed:.2f}%\n")
                out_file.write(f"言い直しの脱落誤り数: {d_deletion_count}, 挿入誤り数: {d_insertion_count}\n")
                out_file.write(f"言い直しの脱落誤り数 (緩和): {d_deletion_count_relaxed}, 挿入誤り数 (緩和): {d_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")
                detail_file.write(f"HYP: {sentence2}")

                detail_file.write("言い直しのマッチング:\n")
                for ref, hyp in d_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif any(ref_word in hyp or hyp_word in ref for ref_word in ref.split() for hyp_word in hyp.split()):
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("\n")

                # テキスト全体のDの数と検出数を更新
                total_d_ref_count += d_ref_count
                total_d_hyp_count += d_hyp_count
                total_d_correct_count += d_correct_count
                total_d_correct_count_relaxed += d_correct_count_relaxed
                total_d_deletion_count += d_deletion_count
                total_d_insertion_count += d_insertion_count
                total_d_deletion_count_relaxed += d_deletion_count_relaxed
                total_d_insertion_count_relaxed += d_insertion_count_relaxed

            # テキスト全体でのDのPrecision、Recall、F1スコアを計算
            total_d_precision = total_d_correct_count / total_d_hyp_count * 100 if total_d_hyp_count > 0 else 0
            total_d_recall = total_d_correct_count / total_d_ref_count * 100 if total_d_ref_count > 0 else 0
            total_d_precision_relaxed = total_d_correct_count_relaxed / total_d_hyp_count * 100 if total_d_hyp_count > 0 else 0
            total_d_recall_relaxed = total_d_correct_count_relaxed / total_d_ref_count * 100 if total_d_ref_count > 0 else 0

            # テキスト全体でのF1スコアを計算
            total_d_f1 = 2 * total_d_precision * total_d_recall / (total_d_precision + total_d_recall) if total_d_precision + total_d_recall > 0 else 0
            total_d_f1_relaxed = 2 * total_d_precision_relaxed * total_d_recall_relaxed / (total_d_precision_relaxed + total_d_recall_relaxed) if total_d_precision_relaxed + total_d_recall_relaxed > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"言い直しの総数: {total_d_ref_count}\n")
            out_file.write(f"言い直しの検出率 - Precision: {total_d_precision:.2f}%, Recall: {total_d_recall:.2f}%, F1: {total_d_f1:.2f}%\n")
            out_file.write(f"言い直しの検出率 (緩和) - Precision: {total_d_precision_relaxed:.2f}%, Recall: {total_d_recall_relaxed:.2f}%, F1: {total_d_f1_relaxed:.2f}%\n")
            out_file.write(f"言い直しの脱落誤り数: {total_d_deletion_count}, 挿入誤り数: {total_d_insertion_count}\n")
            out_file.write(f"言い直しの脱落誤り数 (緩和): {total_d_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_d_insertion_count_relaxed}\n")

def align_words(ref_words, hyp_words):
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if any(ref_word in hyp_word or hyp_word in ref_word for ref_word in ref_words[i - 1].split() for hyp_word in hyp_words[j - 1].split()):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        if any(ref_word in hyp_word or hyp_word in ref_word for ref_word in ref_words[i - 1].split() for hyp_word in hyp_words[j - 1].split()):
            alignment.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            alignment.append((ref_words[i - 1], ""))
            i -= 1
        else:
            alignment.append(("", hyp_words[j - 1]))
            j -= 1

    while i > 0:
        alignment.append((ref_words[i - 1], ""))
        i -= 1

    while j > 0:
        alignment.append(("", hyp_words[j - 1]))
        j -= 1

    alignment.reverse()
    return alignment

# 正解文と音声認識文のテキストファイルのパスを指定
text1_path = "HYP/d/text"
text2_path = "REF/d/text"
output_path = "d_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)