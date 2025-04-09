# Improved Filler (F) Detection Rate Calculator
import re

def calculate_detection_rate(text1_path, text2_path, output_path):
    # テキストファイルを読み込む
    with open(text1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()  # REFファイル（正解文）
    with open(text2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()  # HYPファイル（認識文）

    # Fとカッコの間の文字列を抽出する正規表現パターン
    f_pattern = r'\<F\>\s*([^</]+)\s*\</F\>'

    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open("f_detection_rate_detail.txt", 'w', encoding='utf-8') as detail_file:
            # 各文ごとにFの検出率を計算
            sentence_pattern = r'(\w+_\w+_\w+_\w+)\s'
            sentences1 = re.split(sentence_pattern, text1)  # REF
            sentences2 = re.split(sentence_pattern, text2)  # HYP

            total_f_ref_count = 0
            total_f_hyp_count = 0
            total_f_correct_count = 0
            total_f_correct_count_relaxed = 0
            total_f_deletion_count = 0
            total_f_insertion_count = 0
            total_f_deletion_count_relaxed = 0
            total_f_insertion_count_relaxed = 0

            for i in range(1, len(sentences1), 2):
                if i + 1 >= len(sentences1) or i + 1 >= len(sentences2):
                    continue  # インデックスエラー防止
                    
                sentence_id = sentences1[i]
                sentence1 = sentences1[i+1]  # REF
                sentence2 = sentences2[i+1]  # HYP

                # 正解文と音声認識文からFの単語を抽出
                f_ref_words = re.findall(f_pattern, sentence1)  # REFからの抽出
                f_hyp_words = re.findall(f_pattern, sentence2)  # HYPからの抽出

                # 正規化：空白を統一し、長音記号の違いを許容
                f_ref_words = [normalize_filler(word) for word in f_ref_words]
                f_hyp_words = [normalize_filler(word) for word in f_hyp_words]

                # 完全一致と緩和判定それぞれで適切なアライメントを取得
                f_alignment = get_alignment(f_ref_words, f_hyp_words, strict=True)
                f_alignment_relaxed = get_alignment(f_ref_words, f_hyp_words, strict=False)

                # 正解数、脱落誤り数、挿入誤り数を計算
                f_ref_count = len(f_ref_words)
                f_hyp_count = len(f_hyp_words)
                
                # 完全一致の計算
                f_correct_count = sum(1 for ref, hyp in f_alignment if ref == hyp and ref and hyp)
                f_deletion_count = sum(1 for ref, hyp in f_alignment if ref and not hyp)
                f_insertion_count = sum(1 for ref, hyp in f_alignment if not ref and hyp)
                
                # 緩和基準の計算
                f_correct_count_relaxed = sum(1 for ref, hyp in f_alignment_relaxed if ref and hyp)
                f_deletion_count_relaxed = sum(1 for ref, hyp in f_alignment_relaxed if ref and not hyp)
                f_insertion_count_relaxed = sum(1 for ref, hyp in f_alignment_relaxed if not ref and hyp)

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
                out_file.write(f"REF: {sentence1}")  # REF（正解文）
                out_file.write(f"HYP: {sentence2}")  # HYP（認識文）
                out_file.write(f"フィラーの検出率 - Precision: {f_precision:.2f}%, Recall: {f_recall:.2f}%, F1: {f_f1:.2f}%\n")
                out_file.write(f"フィラーの検出率 (緩和) - Precision: {f_precision_relaxed:.2f}%, Recall: {f_recall_relaxed:.2f}%, F1: {f_f1_relaxed:.2f}%\n")
                out_file.write(f"フィラーの脱落誤り数: {f_deletion_count}, 挿入誤り数: {f_insertion_count}\n")
                out_file.write(f"フィラーの脱落誤り数 (緩和): {f_deletion_count_relaxed}, 挿入誤り数 (緩和): {f_insertion_count_relaxed}\n")
                out_file.write("\n")

                # 詳細出力用のファイルに書き込む
                detail_file.write(f"{sentence_id}\n")
                detail_file.write(f"REF: {sentence1}")  # REF（正解文）
                detail_file.write(f"HYP: {sentence2}")  # HYP（認識文）

                # 完全一致の詳細
                detail_file.write("フィラーのマッチング (完全一致):\n")
                for ref, hyp in f_alignment:
                    if ref == hyp and ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (不一致)\n")
                    elif ref:
                        detail_file.write(f"  {ref} -- (脱落)\n")
                    elif hyp:
                        detail_file.write(f"  -- {hyp} (挿入)\n")
                
                # 緩和基準の詳細
                detail_file.write("\nフィラーのマッチング (緩和基準):\n")
                for ref, hyp in f_alignment_relaxed:
                    if ref == hyp and ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (完全一致)\n")
                    elif ref and hyp:
                        detail_file.write(f"  {ref} -- {hyp} (部分一致: フィラータグの対応あり)\n")
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


def normalize_filler(text):
    """フィラーのテキストを正規化する関数"""
    # 余分な空白を削除
    text = ' '.join(text.split())
    # 長音記号の正規化（「ー」と「ー」は同じものとして扱う）
    text = re.sub(r'ー+', 'ー', text)
    return text


def levenshtein_distance(s1, s2):
    """レーベンシュタイン距離（編集距離）を計算する関数"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(s1, s2):
    """2つの文字列の類似度を0.0〜1.0の範囲で返す関数"""
    if not s1 or not s2:
        return 0.0
    
    # 編集距離を計算
    distance = levenshtein_distance(s1, s2)
    # 最大長
    max_len = max(len(s1), len(s2))
    # 類似度 = 1 - (編集距離 / 最大長)
    return 1.0 - (distance / max_len)


def get_alignment(ref_words, hyp_words, strict=True):
    """
    アライメントを取得する関数
    strict=True: 完全一致の場合のみマッチング
    strict=False: 内容に関わらずフィラータグの対応があればマッチング
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
text1_path = "REF/f/text"  # 正解文（REF）
text2_path = "HYP/f/text"  # 認識文（HYP）
output_path = "f_detection_rate.txt"

calculate_detection_rate(text1_path, text2_path, output_path)