import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()

    # 笑いと言い誤りの正規表現パターン
    laughter_pattern = r'\(L\)\s*([^(/]+)\s*\(/L\)'
    mispronunciation_pattern = r'\(W\)\s*([^(/]+)\s*\(/W\)'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("l_w_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとに笑いと言い誤りの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)
            sentences2 = re.split(sentence_pattern, text2)

            total_laughter_ref_count = 0
            total_mispronunciation_ref_count = 0
            total_laughter_hyp_count = 0
            total_mispronunciation_hyp_count = 0
            total_laughter_correct_count = 0
            total_mispronunciation_correct_count = 0
            total_laughter_correct_count_relaxed = 0
            total_mispronunciation_correct_count_relaxed = 0
            total_laughter_deletion_count = 0
            total_mispronunciation_deletion_count = 0
            total_laughter_insertion_count = 0
            total_mispronunciation_insertion_count = 0
            total_laughter_deletion_count_relaxed = 0
            total_mispronunciation_deletion_count_relaxed = 0
            total_laughter_insertion_count_relaxed = 0
            total_mispronunciation_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]
                sentence2 = sentences2[i+1]

                # 正解文と音声認識文から笑いと言い誤りを抽出
                laughter_ref_words = re.findall(laughter_pattern, sentence1)
                mispronunciation_ref_words = re.findall(mispronunciation_pattern, sentence1)
                laughter_hyp_words = re.findall(laughter_pattern, sentence2)
                mispronunciation_hyp_words = re.findall(mispronunciation_pattern, sentence2)

                # DPを使ってアライメントを取得
                laughter_alignment = align_words(laughter_ref_words, laughter_hyp_words)
                mispronunciation_alignment = align_words(mispronunciation_ref_words, mispronunciation_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                laughter_ref_count = len(laughter_ref_words)
                mispronunciation_ref_count = len(mispronunciation_ref_words)
                laughter_hyp_count = len(laughter_hyp_words)
                mispronunciation_hyp_count = len(mispronunciation_hyp_words)
                laughter_correct_count = sum(1 for ref, hyp in laughter_alignment if ref == hyp)
                mispronunciation_correct_count = sum(1 for ref, hyp in mispronunciation_alignment if ref == hyp)
                laughter_correct_count_relaxed = sum(1 for ref, hyp in laughter_alignment if ref and hyp)
                mispronunciation_correct_count_relaxed = sum(1 for ref, hyp in mispronunciation_alignment if ref and hyp)
                laughter_deletion_count = laughter_ref_count - laughter_correct_count
                mispronunciation_deletion_count = mispronunciation_ref_count - mispronunciation_correct_count
                laughter_insertion_count = laughter_hyp_count - laughter_correct_count
                mispronunciation_insertion_count = mispronunciation_hyp_count - mispronunciation_correct_count
                laughter_deletion_count_relaxed = laughter_ref_count - laughter_correct_count_relaxed
                mispronunciation_deletion_count_relaxed = mispronunciation_ref_count - mispronunciation_correct_count_relaxed
                laughter_insertion_count_relaxed = laughter_hyp_count - laughter_correct_count_relaxed
                mispronunciation_insertion_count_relaxed = mispronunciation_hyp_count - mispronunciation_correct_count_relaxed

                # Precision、Recall、F1スコアを計算
                laughter_precision = laughter_correct_count / laughter_hyp_count * 100 if laughter_hyp_count > 0 else 0
                mispronunciation_precision = mispronunciation_correct_count / mispronunciation_hyp_count * 100 if mispronunciation_hyp_count > 0 else 0
                laughter_recall = laughter_correct_count / laughter_ref_count * 100 if laughter_ref_count > 0 else 0
                mispronunciation_recall = mispronunciation_correct_count / mispronunciation_ref_count * 100 if mispronunciation_ref_count > 0 else 0
                laughter_precision_relaxed = laughter_correct_count_relaxed / laughter_hyp_count * 100 if laughter_hyp_count > 0 else 0
                mispronunciation_precision_relaxed = mispronunciation_correct_count_relaxed / mispronunciation_hyp_count * 100 if mispronunciation_hyp_count > 0 else 0
                laughter_recall_relaxed = laughter_correct_count_relaxed / laughter_ref_count * 100 if laughter_ref_count > 0 else 0
                mispronunciation_recall_relaxed = mispronunciation_correct_count_relaxed / mispronunciation_ref_count * 100 if mispronunciation_ref_count > 0 else 0
                laughter_f1 = 2 * laughter_precision * laughter_recall / (laughter_precision + laughter_recall) if laughter_precision + laughter_recall > 0 else 0
                mispronunciation_f1 = 2 * mispronunciation_precision * mispronunciation_recall / (mispronunciation_precision + mispronunciation_recall) if mispronunciation_precision + mispronunciation_recall > 0 else 0
                laughter_f1_relaxed = 2 * laughter_precision_relaxed * laughter_recall_relaxed / (laughter_precision_relaxed + laughter_recall_relaxed) if laughter_precision_relaxed + laughter_recall_relaxed > 0 else 0
                mispronunciation_f1_relaxed = 2 * mispronunciation_precision_relaxed * mispronunciation_recall_relaxed / (mispronunciation_precision_relaxed + mispronunciation_recall_relaxed) if mispronunciation_precision_relaxed + mispronunciation_recall_relaxed > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {sentence1}")
                out_file.write(f"HYP: {sentence2}")
                out_file.write(f"笑いの検出率 - Precision: {laughter_precision:.2f}%, Recall: {laughter_recall:.2f}%, F1: {laughter_f1:.2f}%\n")
                out_file.write(f"言い誤りの検出率 - Precision: {mispronunciation_precision:.2f}%, Recall: {mispronunciation_recall:.2f}%, F1: {mispronunciation_f1:.2f}%\n")
                out_file.write(f"笑いの検出率 (緩和) - Precision: {laughter_precision_relaxed:.2f}%, Recall: {laughter_recall_relaxed:.2f}%, F1: {laughter_f1_relaxed:.2f}%\n")
                out_file.write(f"言い誤りの検出率 (緩和) - Precision: {mispronunciation_precision_relaxed:.2f}%, Recall: {mispronunciation_recall_relaxed:.2f}%, F1: {mispronunciation_f1_relaxed:.2f}%\n")
                out_file.write(f"笑いの脱落誤り数: {laughter_deletion_count}, 挿入誤り数: {laughter_insertion_count}\n")
                out_file.write(f"言い誤りの脱落誤り数: {mispronunciation_deletion_count}, 挿入誤り数: {mispronunciation_insertion_count}\n")
                out_file.write(f"笑いの脱落誤り数 (緩和): {laughter_deletion_count_relaxed}, 挿入誤り数 (緩和): {laughter_insertion_count_relaxed}\n")
                out_file.write(f"言い誤りの脱落誤り数 (緩和): {mispronunciation_deletion_count_relaxed}, 挿入誤り数 (緩和): {mispronunciation_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")
                detail_file.write(f"HYP: {sentence2}")

                detail_file.write("笑いのマッチング:\n")
                for ref, hyp in laughter_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("言い誤りのマッチング:\n")
                for ref, hyp in mispronunciation_alignment:
                    if ref == hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")

                detail_file.write("\n")

                # テキスト全体の笑いと言い誤りの数と検出数を更新
                total_laughter_ref_count += laughter_ref_count
                total_mispronunciation_ref_count += mispronunciation_ref_count
                total_laughter_hyp_count += laughter_hyp_count
                total_mispronunciation_hyp_count += mispronunciation_hyp_count
                total_laughter_correct_count += laughter_correct_count
                total_mispronunciation_correct_count += mispronunciation_correct_count
                total_laughter_correct_count_relaxed += laughter_correct_count_relaxed
                total_mispronunciation_correct_count_relaxed += mispronunciation_correct_count_relaxed
                total_laughter_deletion_count += laughter_deletion_count
                total_mispronunciation_deletion_count += mispronunciation_deletion_count
                total_laughter_insertion_count += laughter_insertion_count
                total_mispronunciation_insertion_count += mispronunciation_insertion_count
                total_laughter_deletion_count_relaxed += laughter_deletion_count_relaxed
                total_mispronunciation_deletion_count_relaxed += mispronunciation_deletion_count_relaxed
                total_laughter_insertion_count_relaxed += laughter_insertion_count_relaxed
                total_mispronunciation_insertion_count_relaxed += mispronunciation_insertion_count_relaxed

                # テキスト全体での笑いと言い誤りのPrecision、Recall、F1スコアを計算
            total_laughter_precision = total_laughter_correct_count / total_laughter_hyp_count * 100 if total_laughter_hyp_count > 0 else 0
            total_mispronunciation_precision = total_mispronunciation_correct_count / total_mispronunciation_hyp_count * 100 if total_mispronunciation_hyp_count > 0 else 0
            total_laughter_recall = total_laughter_correct_count / total_laughter_ref_count * 100 if total_laughter_ref_count > 0 else 0
            total_mispronunciation_recall = total_mispronunciation_correct_count / total_mispronunciation_ref_count * 100 if total_mispronunciation_ref_count > 0 else 0
            total_laughter_precision_relaxed = total_laughter_correct_count_relaxed / total_laughter_hyp_count * 100 if total_laughter_hyp_count > 0 else 0
            total_mispronunciation_precision_relaxed = total_mispronunciation_correct_count_relaxed / total_mispronunciation_hyp_count * 100 if total_mispronunciation_hyp_count > 0 else 0
            total_laughter_recall_relaxed = total_laughter_correct_count_relaxed / total_laughter_ref_count * 100 if total_laughter_ref_count > 0 else 0
            total_mispronunciation_recall_relaxed = total_mispronunciation_correct_count_relaxed / total_mispronunciation_ref_count * 100 if total_mispronunciation_ref_count > 0 else 0
            total_laughter_f1 = 2 * total_laughter_precision * total_laughter_recall / (total_laughter_precision + total_laughter_recall) if total_laughter_precision + total_laughter_recall > 0 else 0
            total_mispronunciation_f1 = 2 * total_mispronunciation_precision * total_mispronunciation_recall / (total_mispronunciation_precision + total_mispronunciation_recall) if total_mispronunciation_precision + total_mispronunciation_recall > 0 else 0
            total_laughter_f1_relaxed = 2 * total_laughter_precision_relaxed * total_laughter_recall_relaxed / (total_laughter_precision_relaxed + total_laughter_recall_relaxed) if total_laughter_precision_relaxed + total_laughter_recall_relaxed > 0 else 0
            total_mispronunciation_f1_relaxed = 2 * total_mispronunciation_precision_relaxed * total_mispronunciation_recall_relaxed / (total_mispronunciation_precision_relaxed + total_mispronunciation_recall_relaxed) if total_mispronunciation_precision_relaxed + total_mispronunciation_recall_relaxed > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"笑いの総数: {total_laughter_ref_count}, 言い誤りの総数: {total_mispronunciation_ref_count}\n")
            out_file.write(f"笑いの検出率 - Precision: {total_laughter_precision:.2f}%, Recall: {total_laughter_recall:.2f}%, F1: {total_laughter_f1:.2f}%\n")
            out_file.write(f"言い誤りの検出率 - Precision: {total_mispronunciation_precision:.2f}%, Recall: {total_mispronunciation_recall:.2f}%, F1: {total_mispronunciation_f1:.2f}%\n")
            out_file.write(f"笑いの検出率 (緩和) - Precision: {total_laughter_precision_relaxed:.2f}%, Recall: {total_laughter_recall_relaxed:.2f}%, F1: {total_laughter_f1_relaxed:.2f}%\n")
            out_file.write(f"言い誤りの検出率 (緩和) - Precision: {total_mispronunciation_precision_relaxed:.2f}%, Recall: {total_mispronunciation_recall_relaxed:.2f}%, F1: {total_mispronunciation_f1_relaxed:.2f}%\n")
            out_file.write(f"笑いの脱落誤り数: {total_laughter_deletion_count}, 挿入誤り数: {total_laughter_insertion_count}\n")
            out_file.write(f"言い誤りの脱落誤り数: {total_mispronunciation_deletion_count}, 挿入誤り数: {total_mispronunciation_insertion_count}\n")
            out_file.write(f"笑いの脱落誤り数 (緩和): {total_laughter_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_laughter_insertion_count_relaxed}\n")
            out_file.write(f"言い誤りの脱落誤り数 (緩和): {total_mispronunciation_deletion_count_relaxed}, 挿入誤り数 (緩和): {total_mispronunciation_insertion_count_relaxed}\n")

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
text1_path = "REF/l_w/text"
text2_path = "HYP/l_w/text"
output_path = "l_w_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)