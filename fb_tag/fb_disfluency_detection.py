import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()

    # フィラーと言い直しの正規表現パターン
    filler_pattern = r'\(F\)\s*([^(/]+)\s*\(/F\)'
    disfluency_pattern = r'\(D\)\s*([^(/]+)\s*\(/D\)'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにフィラーと言い直しの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)
            sentences2 = re.split(sentence_pattern, text2)

            total_filler_ref_count = 0
            total_disfluency_ref_count = 0
            total_filler_hyp_count = 0
            total_disfluency_hyp_count = 0
            total_filler_correct_count = 0
            total_disfluency_correct_count = 0
            total_filler_correct_count_relaxed = 0
            total_disfluency_correct_count_relaxed = 0
            total_filler_deletion_count = 0
            total_disfluency_deletion_count = 0
            total_filler_insertion_count = 0
            total_disfluency_insertion_count = 0
            total_filler_deletion_count_relaxed = 0
            total_disfluency_deletion_count_relaxed = 0
            total_filler_insertion_count_relaxed = 0
            total_disfluency_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]
                sentence2 = sentences2[i+1]

                # 正解文と音声認識文からフィラーと言い直しを抽出
                filler_ref_words = re.findall(filler_pattern, sentence1)
                disfluency_ref_words = re.findall(disfluency_pattern, sentence1)
                filler_hyp_words = re.findall(filler_pattern, sentence2)
                disfluency_hyp_words = re.findall(disfluency_pattern, sentence2)

                # DPを使ってアライメントを取得
                filler_alignment = align_words(filler_ref_words, filler_hyp_words)
                disfluency_alignment = align_words(disfluency_ref_words, disfluency_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                filler_ref_count = len(filler_ref_words)
                disfluency_ref_count = len(disfluency_ref_words)
                filler_hyp_count = len(filler_hyp_words)
                disfluency_hyp_count = len(disfluency_hyp_words)
                filler_correct_count = sum(1 for ref, hyp in filler_alignment if ref == hyp)
                disfluency_correct_count = sum(1 for ref, hyp in disfluency_alignment if ref == hyp)
                filler_correct_count_relaxed = sum(1 for ref, hyp in filler_alignment if ref and hyp)
                disfluency_correct_count_relaxed = sum(1 for ref, hyp in disfluency_alignment if ref and hyp)
                filler_deletion_count = filler_ref_count - filler_correct_count
                disfluency_deletion_count = disfluency_ref_count - disfluency_correct_count
                filler_insertion_count = filler_hyp_count - filler_correct_count
                disfluency_insertion_count = disfluency_hyp_count - disfluency_correct_count
                filler_deletion_count_relaxed = filler_ref_count - filler_correct_count_relaxed
                disfluency_deletion_count_relaxed = disfluency_ref_count - disfluency_correct_count_relaxed
                filler_insertion_count_relaxed = filler_hyp_count - filler_correct_count_relaxed
                disfluency_insertion_count_relaxed = disfluency_hyp_count - disfluency_correct_count_relaxed

                # Precision、Recall、F1スコアを計算
                filler_precision = filler_correct_count / filler_hyp_count * 100 if filler_hyp_count > 0 else 0
                disfluency_precision = disfluency_correct_count / disfluency_hyp_count * 100 if disfluency_hyp_count > 0 else 0
                filler_recall = filler_correct_count / filler_ref_count * 100 if filler_ref_count > 0 else 0
                disfluency_recall = disfluency_correct_count / disfluency_ref_count * 100 if disfluency_ref_count > 0 else 0
                filler_precision_relaxed = filler_correct_count_relaxed / filler_hyp_count * 100 if filler_hyp_count > 0 else 0
                disfluency_precision_relaxed = disfluency_correct_count_relaxed / disfluency_hyp_count * 100 if disfluency_hyp_count > 0 else 0
                filler_recall_relaxed = filler_correct_count_relaxed / filler_ref_count * 100 if filler_ref_count > 0 else 0
                disfluency_recall_relaxed = disfluency_correct_count_relaxed / disfluency_ref_count * 100 if disfluency_ref_count > 0 else 0
                filler_f1 = 2 * filler_precision * filler_recall / (filler_precision + filler_recall) if filler_precision + filler_recall > 0 else 0
                disfluency_f1 = 2 * disfluency_precision * disfluency_recall / (disfluency_precision + disfluency_recall) if disfluency_precision + disfluency_recall > 0 else 0
                filler_f1_relaxed = 2 * filler_precision_relaxed * filler_recall_relaxed / (filler_precision_relaxed + filler_recall_relaxed) if filler_precision_relaxed + filler_recall_relaxed > 0 else 0
                disfluency_f1_relaxed = 2 * disfluency_precision_relaxed * disfluency_recall_relaxed / (disfluency_precision_relaxed + disfluency_recall_relaxed) if disfluency_precision_relaxed + disfluency_recall_relaxed > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {sentence1}")
                out_file.write(f"HYP: {sentence2}")
                out_file.write(f"フィラーの検出率 - Precision: {filler_precision:.2f}%, Recall: {filler_recall:.2f}%, F1: {filler_f1:.2f}%\n")
                out_file.write(f"言い直しの検出率 - Precision: {disfluency_precision:.2f}%, Recall: {disfluency_recall:.2f}%, F1: {disfluency_f1:.2f}%\n")
                out_file.write(f"フィラーの検出率 (緩和) - Precision: {filler_precision_relaxed:.2f}%, Recall: {filler_recall_relaxed:.2f}%, F1: {filler_f1_relaxed:.2f}%\n")
                out_file.write(f"言い直しの検出率 (緩和) - Precision: {disfluency_precision_relaxed:.2f}%, Recall: {disfluency_recall_relaxed:.2f}%, F1: {disfluency_f1_relaxed:.2f}%\n")
                out_file.write(f"フィラーの脱落誤り数: {filler_deletion_count}, 挿入誤り数: {filler_insertion_count}\n")
                out_file.write(f"言い直しの脱落誤り数: {disfluency_deletion_count}, 挿入誤り数: {disfluency_insertion_count}\n")
                out_file.write(f"フィラーの脱落誤り数 (緩和): {filler_deletion_count_relaxed}, 挿入誤り数 (緩和): {filler_insertion_count_relaxed}\n")
                out_file.write(f"言い直しの脱落誤り数 (緩和): {disfluency_deletion_count_relaxed}, 挿入誤り数 (緩和): {disfluency_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")
                detail_file.write(f"HYP: {sentence2}")

                detail_file.write("フィラーのマッチング:\n")
                for ref, hyp in filler_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("言い直しのマッチング:\n")
                for ref, hyp in disfluency_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("\n")

                # テキスト全体のフィラーと言い直しの数と検出数を更新
                total_filler_ref_count += filler_ref_count
                total_disfluency_ref_count += disfluency_ref_count
                total_filler_hyp_count += filler_hyp_count
                total_disfluency_hyp_count += disfluency_hyp_count
                total_filler_correct_count += filler_correct_count
                total_disfluency_correct_count += disfluency_correct_count
                total_filler_correct_count_relaxed += filler_correct_count_relaxed
                total_disfluency_correct_count_relaxed += disfluency_correct_count_relaxed
                total_filler_deletion_count += filler_deletion_count
                total_disfluency_deletion_count += disfluency_deletion_count
                total_filler_insertion_count += filler_insertion_count
                total_disfluency_insertion_count += disfluency_insertion_count
                total_filler_deletion_count_relaxed += filler_deletion_count_relaxed
                total_disfluency_deletion_count_relaxed += disfluency_deletion_count_relaxed
                total_filler_insertion_count_relaxed += filler_insertion_count_relaxed
                total_disfluency_insertion_count_relaxed += disfluency_insertion_count_relaxed

            # テキスト全体でのフィラーと言い直しのPrecision、Recall、F1スコアを計算
            total_filler_precision = total_filler_correct_count / total_filler_hyp_count * 100 if total_filler_hyp_count > 0 else 0
            total_disfluency_precision = total_disfluency_correct_count / total_disfluency_hyp_count * 100 if total_disfluency_hyp_count > 0 else 0
            total_filler_recall = total_filler_correct_count / total_filler_ref_count * 100 if total_filler_ref_count > 0 else 0
            total_disfluency_recall = total_disfluency_correct_count / total_disfluency_ref_count * 100 if total_disfluency_ref_count > 0 else 0
            total_filler_precision_relaxed = total_filler_correct_count_relaxed / total_filler_hyp_count * 100 if total_filler_hyp_count > 0 else 0
            total_disfluency_precision_relaxed = total_disfluency_correct_count_relaxed / total_disfluency_hyp_count * 100 if total_disfluency_hyp_count > 0 else 0
            total_filler_recall_relaxed = total_filler_correct_count_relaxed / total_filler_ref_count * 100 if total_filler_ref_count > 0 else 0
            total_disfluency_recall_relaxed = total_disfluency_correct_count_relaxed / total_disfluency_ref_count * 100 if total_disfluency_ref_count > 0 else 0
            total_filler_f1 = 2 * total_filler_precision * total_filler_recall / (total_filler_precision + total_filler_recall) if total_filler_precision + total_filler_recall > 0 else 0
            total_disfluency_f1 = 2 * total_disfluency_precision * total_disfluency_recall / (total_disfluency_precision + total_disfluency_recall) if total_disfluency_precision + total_disfluency_recall > 0 else 0
            total_filler_f1_relaxed = 2 * total_filler_precision_relaxed * total_filler_recall_relaxed / (total_filler_precision_relaxed + total_filler_recall_relaxed) if total_filler_precision_relaxed + total_filler_recall_relaxed > 0 else 0
            total_disfluency_f1_relaxed = 2 * total_disfluency_precision_relaxed * total_disfluency_recall_relaxed / (total_disfluency_precision_relaxed + total_disfluency_recall_relaxed) if total_disfluency_precision_relaxed + total_disfluency_recall_relaxed > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"フィラーの総数: {total_filler_ref_count}, 言い直しの総数: {total_disfluency_ref_count}\n")
            out_file.write(f"フィラーの検出率 - Precision: {total_filler_precision:.2f}%, Recall: {total_filler_recall:.2f}%, F1: {total_filler_f1:.2f}%\n")
            out_file.write(f"言い直しの検出率 - Precision: {total_disfluency_precision:.2f}%, Recall: {total_disfluency_recall:.2f}%, F1: {total_disfluency_f1:.2f}%\n")
            out_file.write(f"フィラーの検出率 (緩和) - Precision: {total_filler_precision_relaxed:.2f}%, Recall: {total_filler_recall_relaxed:.2f}%, F1: {total_filler_f1_relaxed:.2f}%\n")
            out_file.write(f"言い直しの検出率 (緩和) - Precision: {total_disfluency_precision_relaxed:.2f}%, Recall: {total_disfluency_recall_relaxed:.2f}%, F1: {total_disfluency_f1_relaxed:.2f}%\n")
            out_file.write(f"フィラーの脱落誤り数: {total_filler_deletion_count}, 挿入誤り数: {total_filler_insertion_count}\n")
            out_file.write(f"言い直しの脱落誤り数: {total_disfluency_deletion_count}, 挿入誤り数: {total_disfluency_insertion_count}\n")
            out_file.write(f"フィラーの脱落誤り数 (緩和): {total_filler_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_filler_insertion_count_relaxed}\n")
            out_file.write(f"言い直しの脱落誤り数 (緩和): {total_disfluency_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_disfluency_insertion_count_relaxed}\n")

def align_words(ref_words, hyp_words):
   n = len(ref_words)
   m = len(hyp_words)
   dp = [[0] * (m + 1) for _ in range(n + 1)]

   for i in range(1, n + 1):
       for j in range(1, m + 1):
           if ref_words[i - 1] == hyp_words[j - 1]:
               dp[i][j] = dp[i - 1][j - 1] + 1
           elif any(ref_word in hyp_words[j - 1] or hyp_word in ref_words[i - 1] for ref_word in ref_words[i - 1].split() for hyp_word in hyp_words[j - 1].split()):
               dp[i][j] = dp[i - 1][j - 1] + 0.5
           else:
               dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

   alignment = []
   i, j = n, m
   while i > 0 and j > 0:
       if ref_words[i - 1] == hyp_words[j - 1]:
           alignment.append((ref_words[i - 1], hyp_words[j - 1]))
           i -= 1
           j -= 1
       elif any(ref_word in hyp_words[j - 1] or hyp_word in ref_words[i - 1] for ref_word in ref_words[i - 1].split() for hyp_word in hyp_words[j - 1].split()):
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
text1_path = "REF/text"
text2_path = "HYP/text"
output_path = "detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)