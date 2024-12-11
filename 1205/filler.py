import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()

    # Fとカッコの間の文字列を抽出する正規表現パターン
    f_pattern = r'\(F\s+([^)]+)\)'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("f_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにFの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)
            sentences2 = re.split(sentence_pattern, text2)

            total_f_ref_count = 0
            total_f_hyp_count = 0
            total_f_correct_count = 0
            total_f_correct_count_relaxed = 0
            total_f_deletion_count = 0
            total_f_insertion_count = 0
            total_f_deletion_count_relaxed = 0
            total_f_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]
                sentence2 = sentences2[i+1]

                # 正解文と音声認識文からFの単語を抽出
                f_ref_words = re.findall(f_pattern, sentence1)
                f_hyp_words = re.findall(f_pattern, sentence2)

                # DPを使ってアライメントを取得
                f_alignment = align_words(f_ref_words, f_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                f_ref_count = len(f_ref_words)
                f_hyp_count = len(f_hyp_words)
                f_correct_count = sum(1 for ref, hyp in f_alignment if ref == hyp)
                f_correct_count_relaxed = sum(1 for ref, hyp in f_alignment if ref and hyp)
                f_deletion_count = f_ref_count - f_correct_count
                f_insertion_count = f_hyp_count - f_correct_count
                f_deletion_count_relaxed = f_ref_count - f_correct_count_relaxed
                f_insertion_count_relaxed = f_hyp_count - f_correct_count_relaxed

                # Precision、Recall、F1スコアを計算
                f_precision = f_correct_count / f_hyp_count * 100 if f_hyp_count > 0 else 0
                f_recall = f_correct_count / f_ref_count * 100 if f_ref_count > 0 else 0
                f_precision_relaxed = f_correct_count_relaxed / f_hyp_count * 100 if f_hyp_count > 0 else 0
                f_recall_relaxed = f_correct_count_relaxed / f_ref_count * 100 if f_ref_count > 0 else 0

                # F1スコアを計算
                f_f1 = 2 * f_precision * f_recall / (f_precision + f_recall) if f_precision + f_recall > 0 else 0
                f_f1_relaxed = 2 * f_precision_relaxed * f_recall_relaxed / (f_precision_relaxed + f_recall_relaxed) if f_precision_relaxed + f_recall_relaxed > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {sentence1}")
                out_file.write(f"HYP: {sentence2}")
                out_file.write(f"フィラーの検出率 - Precision: {f_precision:.2f}%, Recall: {f_recall:.2f}%, F1: {f_f1:.2f}%\n")
                out_file.write(f"フィラーの検出率 (緩和) - Precision: {f_precision_relaxed:.2f}%, Recall: {f_recall_relaxed:.2f}%, F1: {f_f1_relaxed:.2f}%\n")
                out_file.write(f"フィラーの脱落誤り数: {f_deletion_count}, 挿入誤り数: {f_insertion_count}\n")
                out_file.write(f"フィラーの脱落誤り数 (緩和): {f_deletion_count_relaxed}, 挿入誤り数 (緩和): {f_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")
                detail_file.write(f"HYP: {sentence2}")

                detail_file.write("フィラーのマッチング:\n")
                for ref, hyp in f_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif any(ref_word in hyp or hyp_word in ref for ref_word in ref.split() for hyp_word in hyp.split()):
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("\n")

                # テキスト全体のFの数と検出数を更新
                total_f_ref_count += f_ref_count
                total_f_hyp_count += f_hyp_count
                total_f_correct_count += f_correct_count
                total_f_correct_count_relaxed += f_correct_count_relaxed
                total_f_deletion_count += f_deletion_count
                total_f_insertion_count += f_insertion_count
                total_f_deletion_count_relaxed += f_deletion_count_relaxed
                total_f_insertion_count_relaxed += f_insertion_count_relaxed

            # テキスト全体でのFのPrecision、Recall、F1スコアを計算
            total_f_precision = total_f_correct_count / total_f_hyp_count * 100 if total_f_hyp_count > 0 else 0
            total_f_recall = total_f_correct_count / total_f_ref_count * 100 if total_f_ref_count > 0 else 0
            total_f_precision_relaxed = total_f_correct_count_relaxed / total_f_hyp_count * 100 if total_f_hyp_count > 0 else 0
            total_f_recall_relaxed = total_f_correct_count_relaxed / total_f_ref_count * 100 if total_f_ref_count > 0 else 0

            # テキスト全体でのF1スコアを計算
            total_f_f1 = 2 * total_f_precision * total_f_recall / (total_f_precision + total_f_recall) if total_f_precision + total_f_recall > 0 else 0
            total_f_f1_relaxed = 2 * total_f_precision_relaxed * total_f_recall_relaxed / (total_f_precision_relaxed + total_f_recall_relaxed) if total_f_precision_relaxed + total_f_recall_relaxed > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"フィラーの総数: {total_f_ref_count}\n")
            out_file.write(f"フィラーの検出率 - Precision: {total_f_precision:.2f}%, Recall: {total_f_recall:.2f}%, F1: {total_f_f1:.2f}%\n")
            out_file.write(f"フィラーの検出率 (緩和) - Precision: {total_f_precision_relaxed:.2f}%, Recall: {total_f_recall_relaxed:.2f}%, F1: {total_f_f1_relaxed:.2f}%\n")
            out_file.write(f"フィラーの脱落誤り数: {total_f_deletion_count}, 挿入誤り数: {total_f_insertion_count}\n")
            out_file.write(f"フィラーの脱落誤り数 (緩和): {total_f_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_f_insertion_count_relaxed}\n")

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
text1_path = "HYP/f/text"
text2_path = "REF/f/text"
output_path = "f_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)