import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()  # REFファイル（正解文）
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()  # HYPファイル（認識文）

    # Rとカッコの間の文字列を抽出する正規表現パターン
    r_pattern = r'\(([^)]+)R\)'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("r_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにRの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)  # REF
            sentences2 = re.split(sentence_pattern, text2)  # HYP

            total_r_ref_count = 0
            total_r_hyp_count = 0
            total_r_correct_count = 0
            total_r_correct_count_relaxed = 0
            total_r_deletion_count = 0
            total_r_insertion_count = 0
            total_r_deletion_count_relaxed = 0
            total_r_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]  # REF
                sentence2 = sentences2[i+1]  # HYP

                # 正解文と音声認識文からRの単語を抽出
                r_ref_words = re.findall(r_pattern, sentence1)  # REFからの抽出
                r_hyp_words = re.findall(r_pattern, sentence2)  # HYPからの抽出

                # DPを使ってアライメントを取得
                r_alignment = align_words(r_ref_words, r_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                r_ref_count = len(r_ref_words)
                r_hyp_count = len(r_hyp_words)
                r_correct_count = sum(1 for ref, hyp in r_alignment if ref == hyp)
                r_correct_count_relaxed = sum(1 for ref, hyp in r_alignment if ref and hyp)
                r_deletion_count = r_ref_count - r_correct_count
                r_insertion_count = r_hyp_count - r_correct_count
                r_deletion_count_relaxed = r_ref_count - r_correct_count_relaxed
                r_insertion_count_relaxed = r_hyp_count - r_correct_count_relaxed

                # Precision、Recall、F1スコアを計算
                r_precision = r_correct_count / r_hyp_count * 100 if r_hyp_count > 0 else 0
                r_recall = r_correct_count / r_ref_count * 100 if r_ref_count > 0 else 0
                r_precision_relaxed = r_correct_count_relaxed / r_hyp_count * 100 if r_hyp_count > 0 else 0
                r_recall_relaxed = r_correct_count_relaxed / r_ref_count * 100 if r_ref_count > 0 else 0

                # F1スコアを計算
                r_f1 = 2 * r_precision * r_recall / (r_precision + r_recall) if r_precision + r_recall > 0 else 0
                r_f1_relaxed = 2 * r_precision_relaxed * r_recall_relaxed / (r_precision_relaxed + r_recall_relaxed) if r_precision_relaxed + r_recall_relaxed > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {sentence1}")  # REF（正解文）
                out_file.write(f"HYP: {sentence2}")  # HYP（認識文）
                out_file.write(f"感動詞（R）の検出率 - Precision: {r_precision:.2f}%, Recall: {r_recall:.2f}%, F1: {r_f1:.2f}%\n")
                out_file.write(f"感動詞（R）の検出率 (緩和) - Precision: {r_precision_relaxed:.2f}%, Recall: {r_recall_relaxed:.2f}%, F1: {r_f1_relaxed:.2f}%\n")
                out_file.write(f"感動詞（R）の脱落誤り数: {r_deletion_count}, 挿入誤り数: {r_insertion_count}\n")
                out_file.write(f"感動詞（R）の脱落誤り数 (緩和): {r_deletion_count_relaxed}, 挿入誤り数 (緩和): {r_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")  # REF（正解文）
                detail_file.write(f"HYP: {sentence2}")  # HYP（認識文）

                detail_file.write("感動詞（R）のマッチング:\n")
                for ref, hyp in r_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif any(ref_word in hyp or hyp_word in ref for ref_word in ref.split() for hyp_word in hyp.split()):
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("\n")

                # テキスト全体のRの数と検出数を更新
                total_r_ref_count += r_ref_count
                total_r_hyp_count += r_hyp_count
                total_r_correct_count += r_correct_count
                total_r_correct_count_relaxed += r_correct_count_relaxed
                total_r_deletion_count += r_deletion_count
                total_r_insertion_count += r_insertion_count
                total_r_deletion_count_relaxed += r_deletion_count_relaxed
                total_r_insertion_count_relaxed += r_insertion_count_relaxed

            # テキスト全体でのRのPrecision、Recall、F1スコアを計算
            total_r_precision = total_r_correct_count / total_r_hyp_count * 100 if total_r_hyp_count > 0 else 0
            total_r_recall = total_r_correct_count / total_r_ref_count * 100 if total_r_ref_count > 0 else 0
            total_r_precision_relaxed = total_r_correct_count_relaxed / total_r_hyp_count * 100 if total_r_hyp_count > 0 else 0
            total_r_recall_relaxed = total_r_correct_count_relaxed / total_r_ref_count * 100 if total_r_ref_count > 0 else 0

            # テキスト全体でのF1スコアを計算
            total_r_f1 = 2 * total_r_precision * total_r_recall / (total_r_precision + total_r_recall) if total_r_precision + total_r_recall > 0 else 0
            total_r_f1_relaxed = 2 * total_r_precision_relaxed * total_r_recall_relaxed / (total_r_precision_relaxed + total_r_recall_relaxed) if total_r_precision_relaxed + total_r_recall_relaxed > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"感動詞（R）の総数: {total_r_ref_count}\n")
            out_file.write(f"感動詞（R）の検出率 - Precision: {total_r_precision:.2f}%, Recall: {total_r_recall:.2f}%, F1: {total_r_f1:.2f}%\n")
            out_file.write(f"感動詞（R）の検出率 (緩和) - Precision: {total_r_precision_relaxed:.2f}%, Recall: {total_r_recall_relaxed:.2f}%, F1: {total_r_f1_relaxed:.2f}%\n")
            out_file.write(f"感動詞（R）の脱落誤り数: {total_r_deletion_count}, 挿入誤り数: {total_r_insertion_count}\n")
            out_file.write(f"感動詞（R）の脱落誤り数 (緩和): {total_r_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_r_insertion_count_relaxed}\n")

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

# 正解文と音声認識文のテキストファイルのパスを指定（順序を修正）
text1_path = "REF/r/text"  # 正解文（REF）
text2_path = "HYP/r/text"  # 認識文（HYP）
output_path = "r_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)