# Improved Self-correction (D) Detection Rate Calculator
import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()  # REFファイル（正解文）
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()  # HYPファイル（認識文）

    # Dとカッコの間の文字列を抽出する正規表現パターン
    d_pattern = r'\<D\>\s*([^</]+)\s*\</D\>'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("d_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにDの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)  # REF
            sentences2 = re.split(sentence_pattern, text2)  # HYP

            total_d_ref_count = 0
            total_d_hyp_count = 0
            total_d_correct_count = 0
            total_d_correct_count_relaxed = 0
            total_d_deletion_count = 0
            total_d_insertion_count = 0
            total_d_deletion_count_relaxed = 0
            total_d_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                if i + 1 >= len(sentences1) or i + 1 >= len(sentences2):
                    continue  # インデックスエラー防止
                    
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]  # REF
                sentence2 = sentences2[i+1]  # HYP

                # 正解文と音声認識文からDの単語を抽出
                d_ref_words = re.findall(d_pattern, sentence1)  # REFからの抽出
                d_hyp_words = re.findall(d_pattern, sentence2)  # HYPからの抽出

                # 正規化：空白を統一
                d_ref_words = [normalize_text(word) for word in d_ref_words]
                d_hyp_words = [normalize_text(word) for word in d_hyp_words]

                # 完全一致と緩和判定それぞれで適切なアライメントを取得
                d_alignment = get_alignment(d_ref_words, d_hyp_words, strict=True)
                d_alignment_relaxed = get_alignment(d_ref_words, d_hyp_words, strict=False)

                # 正解数、脱落誤り数、挿入誤り数を計算
                d_ref_count = len(d_ref_words)
                d_hyp_count = len(d_hyp_words)
                
                # 完全一致の計算
                d_correct_count = sum(1 for ref, hyp in d_alignment if ref == hyp and ref and hyp)
                d_deletion_count = sum(1 for ref, hyp in d_alignment if ref and not hyp)
                d_insertion_count = sum(1 for ref, hyp in d_alignment if not ref and hyp)
                
                # 緩和基準の計算
                d_correct_count_relaxed = sum(1 for ref, hyp in d_alignment_relaxed if ref and hyp)
                d_deletion_count_relaxed = sum(1 for ref, hyp in d_alignment_relaxed if ref and not hyp)
                d_insertion_count_relaxed = sum(1 for ref, hyp in d_alignment_relaxed if not ref and hyp)

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
                out_file.write(f"REF: {sentence1}")  # REF（正解文）
                out_file.write(f"HYP: {sentence2}")  # HYP（認識文）
                out_file.write(f"言い直しの検出率 - Precision: {d_precision:.2f}%, Recall: {d_recall:.2f}%, F1: {d_f1:.2f}%\n")
                out_file.write(f"言い直しの検出率 (緩和) - Precision: {d_precision_relaxed:.2f}%, Recall: {d_recall_relaxed:.2f}%, F1: {d_f1_relaxed:.2f}%\n")
                out_file.write(f"言い直しの脱落誤り数: {d_deletion_count}, 挿入誤り数: {d_insertion_count}\n")
                out_file.write(f"言い直しの脱落誤り数 (緩和): {d_deletion_count_relaxed}, 挿入誤り数 (緩和): {d_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")  # REF（正解文）
                detail_file.write(f"HYP: {sentence2}")  # HYP（認識文）

                # 完全一致の詳細
                detail_file.write("言い直しのマッチング (完全一致):\n")
                for ref, hyp in d_alignment:
                    if ref == hyp and ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (不一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")
                
                # 緩和基準の詳細
                detail_file.write("\n言い直しのマッチング (緩和基準):\n")
                for ref, hyp in d_alignment_relaxed:
                    if ref == hyp and ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致: 言い直しタグの対応あり)\n")
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


def normalize_text(text):
    """テキストを正規化する関数"""
    # 余分な空白を削除
    text = ' '.join(text.split())
    return text


def get_alignment(ref_words, hyp_words, strict=True):
    """
    アライメントを取得する関数
    strict=True: 完全一致の場合のみマッチング
    strict=False: 内容に関わらずタグの対応があればマッチング
    """
    n = len(ref_words)
    m = len(hyp_words)
    
    if n == 0 and m == 0:
        return []
    
    # dp行列とバックポインタを初期化
    dp = [[0.0 for _ in range(m+1)] for _ in range(n+1)]
    bp = [[None for _ in range(m+1)] for _ in range(n+1)]
    
    # 動的計画法でスコアを計算
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 and j == 0:
                continue
                
            # 削除
            if i > 0:
                score_del = dp[i-1][j]
                if dp[i][j] < score_del or bp[i][j] is None:
                    dp[i][j] = score_del
                    bp[i][j] = (i-1, j, 'del')
            
            # 挿入
            if j > 0:
                score_ins = dp[i][j-1]
                if dp[i][j] < score_ins or bp[i][j] is None:
                    dp[i][j] = score_ins
                    bp[i][j] = (i, j-1, 'ins')
            
            # 置換または一致
            if i > 0 and j > 0:
                # strictモード: 完全一致の場合のみマッチング
                if strict:
                    match_bonus = 2.0 if ref_words[i-1] == hyp_words[j-1] else 0.0
                # 緩和モード: 内容に関わらずマッチング
                else:
                    match_bonus = 1.0
                
                score_sub = dp[i-1][j-1] + match_bonus
                if dp[i][j] < score_sub:
                    dp[i][j] = score_sub
                    bp[i][j] = (i-1, j-1, 'sub')
    
    # バックトレースでアライメントを生成
    alignment = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if bp[i][j] is None:
            break
            
        prev_i, prev_j, op = bp[i][j]
        
        if op == 'sub':  # 置換または一致
            alignment.append((ref_words[i-1], hyp_words[j-1]))
        elif op == 'del':  # 削除
            alignment.append((ref_words[i-1], ""))
        elif op == 'ins':  # 挿入
            alignment.append(("", hyp_words[j-1]))
            
        i, j = prev_i, prev_j
        
    alignment.reverse()
    return alignment


# 正解文と音声認識文のテキストファイルのパスを指定
text1_path = "REF/d/text"  # 正解文（REF）
text2_path = "HYP/d/text"  # 認識文（HYP）
output_path = "d_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)