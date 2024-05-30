import re

def calculate_detection_rate(hyp_path, ref_path, output_path):
    # テキストファイルを読み込む
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyp_text = f.read()
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_text = f.read()

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにFとDの検出率を計算
            sentence_pattern = r'(C\d+_\d+IC\d+_\d+_\d+|K\d+_\d+aIC\d+_\d+_\d+|T\d+_\d+IC\d+_\d+_\d+)\s'
            hyp_sentences = re.split(sentence_pattern, hyp_text)
            ref_sentences = re.split(sentence_pattern, ref_text)

            total_f_ref_count = 0
            total_d_ref_count = 0
            total_f_hyp_count = 0
            total_d_hyp_count = 0
            total_f_correct_count = 0
            total_d_correct_count = 0
            total_f_deletion_count = 0
            total_d_deletion_count = 0
            total_f_insertion_count = 0
            total_d_insertion_count = 0

            i = 1
            while i < len(hyp_sentences) and i < len(ref_sentences):
                sentence_id = hyp_sentences[i]
                hyp_sentence = hyp_sentences[i+1] if i+1 < len(hyp_sentences) else ""
                ref_sentence = ref_sentences[i+1] if i+1 < len(ref_sentences) else ""

                # 正解文と音声認識文からFとDの単語を抽出
                f_ref_words = re.findall(r'\+F', ref_sentence)
                d_ref_words = re.findall(r'\+D', ref_sentence)
                f_hyp_words = re.findall(r'\+F', hyp_sentence)
                d_hyp_words = re.findall(r'\+D', hyp_sentence)

                # DPを使ってアライメントを取得
                f_alignment = align_words(f_ref_words, f_hyp_words)
                d_alignment = align_words(d_ref_words, d_hyp_words)

                # 正解数、脱落誤り数、挿入誤り数を計算
                f_ref_count = len(f_ref_words)
                d_ref_count = len(d_ref_words)
                f_hyp_count = len(f_hyp_words)
                d_hyp_count = len(d_hyp_words)
                f_correct_count = sum(1 for ref, hyp in f_alignment if ref == hyp)
                d_correct_count = sum(1 for ref, hyp in d_alignment if ref == hyp)
                f_deletion_count = f_ref_count - f_correct_count
                d_deletion_count = d_ref_count - d_correct_count
                f_insertion_count = f_hyp_count - f_correct_count
                d_insertion_count = d_hyp_count - d_correct_count

                # Precision、Recallを計算
                f_precision = f_correct_count / f_hyp_count * 100 if f_hyp_count > 0 else 0
                d_precision = d_correct_count / d_hyp_count * 100 if d_hyp_count > 0 else 0
                f_recall = f_correct_count / f_ref_count * 100 if f_ref_count > 0 else 0
                d_recall = d_correct_count / d_ref_count * 100 if d_ref_count > 0 else 0

                # 結果を出力ファイルに書き込む
                out_file.write(f"{sentence_id}\n")
                out_file.write(f"REF: {ref_sentence}")
                out_file.write(f"HYP: {hyp_sentence}")
                out_file.write(f"フィラーの検出率 - Precision: {f_precision:.2f}%, Recall: {f_recall:.2f}%\n")
                out_file.write(f"言い直しの検出率 - Precision: {d_precision:.2f}%, Recall: {d_recall:.2f}%\n")
                out_file.write(f"フィラーの脱落誤り数: {f_deletion_count}, 挿入誤り数: {f_insertion_count}\n")
                out_file.write(f"言い直しの脱落誤り数: {d_deletion_count}, 挿入誤り数: {d_insertion_count}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {ref_sentence}")
                detail_file.write(f"HYP: {hyp_sentence}")

                detail_file.write("フィラーのマッチング:\n")
                for ref, hyp in f_alignment:
                    if ref == hyp:
                        detail_file.write(f"  +F -- +F (完全一致)\n")
                    elif ref:
                        detail_file.write(f"  +F -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- +F (挿入)\n")

                detail_file.write("言い直しのマッチング:\n")
                for ref, hyp in d_alignment:
                    if ref == hyp:
                        detail_file.write(f"  +D -- +D (完全一致)\n")
                    elif ref:
                        detail_file.write(f"  +D -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- +D (挿入)\n")

                detail_file.write("\n")

                # テキスト全体のFとDの数と検出数を更新
                total_f_ref_count += f_ref_count
                total_d_ref_count += d_ref_count
                total_f_hyp_count += f_hyp_count
                total_d_hyp_count += d_hyp_count
                total_f_correct_count += f_correct_count
                total_d_correct_count += d_correct_count
                total_f_deletion_count += f_deletion_count
                total_d_deletion_count += d_deletion_count
                total_f_insertion_count += f_insertion_count
                total_d_insertion_count += d_insertion_count

                i += 2

            # テキスト全体でのFとDのPrecision、Recallを計算
            total_f_precision = total_f_correct_count / total_f_hyp_count * 100 if total_f_hyp_count > 0 else 0
            total_d_precision = total_d_correct_count / total_d_hyp_count * 100 if total_d_hyp_count > 0 else 0
            total_f_recall = total_f_correct_count / total_f_ref_count * 100 if total_f_ref_count > 0 else 0
            total_d_recall = total_d_correct_count / total_d_ref_count * 100 if total_d_ref_count > 0 else 0

            # テキスト全体の検出率を出力ファイルに書き込む
            out_file.write("テキスト全体での検出率:\n")
            out_file.write(f"フィラーの検出率 - Precision: {total_f_precision:.2f}%, Recall: {total_f_recall:.2f}%\n")
            out_file.write(f"言い直しの検出率 - Precision: {total_d_precision:.2f}%, Recall: {total_d_recall:.2f}%\n")
            out_file.write(f"フィラーの脱落誤り数: {total_f_deletion_count}, 挿入誤り数: {total_f_insertion_count}\n")
            out_file.write(f"言い直しの脱落誤り数: {total_d_deletion_count}, 挿入誤り数: {total_d_insertion_count}\n")

def align_words(ref_words, hyp_words):
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        if ref_words[i - 1] == hyp_words[j - 1]:
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
hyp_path = "HYP/text"
ref_path = "REF/text"
output_path = "detection_rate.txt"

calculate_detection_rate(hyp_path, ref_path, output_path)