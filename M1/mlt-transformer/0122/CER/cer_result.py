import numpy as np
from datetime import datetime

def calculate_cer(ref, hyp):
    """
    Calculate Character Error Rate using the same method as ESPnet (SCLITE-like)
    CER = (S + D + I) / N, where:
    S = number of substitutions
    D = number of deletions
    I = number of insertions
    N = number of characters in reference
    
    Characters are space-separated tokens
    """
    # Split by space to get character tokens (same as ESPnet)
    ref_chars = ref.split()
    hyp_chars = hyp.split()
    
    # Create DP matrix
    ref_len = len(ref_chars)
    hyp_len = len(hyp_chars)
    dp = np.zeros((ref_len + 1, hyp_len + 1))
    
    # Initialize first row and column
    for i in range(ref_len + 1):
        dp[i][0] = i  # deletions
    for j in range(hyp_len + 1):
        dp[0][j] = j  # insertions
        
    # Initialize backtrace matrix to track operations
    ops = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)
    ops[:, 0] = 1  # deletion
    ops[0, :] = 2  # insertion
    
    # Fill the matrix
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 0  # correct
            else:
                substitution = dp[i-1][j-1] + 1
                deletion = dp[i-1][j] + 1
                insertion = dp[i][j-1] + 1
                
                dp[i][j] = min(substitution, deletion, insertion)
                
                if dp[i][j] == substitution:
                    ops[i][j] = 3  # substitution
                elif dp[i][j] == deletion:
                    ops[i][j] = 1  # deletion
                else:
                    ops[i][j] = 2  # insertion
    
    # Backtrace to count operations
    i, j = ref_len, hyp_len
    substitutions = deletions = insertions = correct = 0
    
    while i > 0 or j > 0:
        if ops[i][j] == 3:  # substitution
            substitutions += 1
            i -= 1
            j -= 1
        elif ops[i][j] == 1:  # deletion
            deletions += 1
            i -= 1
        elif ops[i][j] == 2:  # insertion
            insertions += 1
            j -= 1
        else:  # correct
            correct += 1
            i -= 1
            j -= 1
    
    # Calculate CER
    if ref_len == 0:
        return 0.0, (correct, substitutions, deletions, insertions)
    
    cer = (substitutions + deletions + insertions) / ref_len
    return cer, (correct, substitutions, deletions, insertions)

def process_recognition_results(ref_lines, hyp_lines):
    """
    Process the recognition results and calculate CER for each utterance (ESPnet-compatible)
    ref_lines: list of lines from REF/text
    hyp_lines: list of lines from HYP/text
    """
    results = []
    total_ref_chars = 0
    total_correct = total_sub = total_del = total_ins = 0
    
    # Create dictionary for hyp lines for easier matching
    hyp_dict = {line.split()[0]: line for line in hyp_lines if line.strip()}
    
    for ref_line in ref_lines:
        if not ref_line.strip():
            continue
            
        # Extract ID and text
        parts = ref_line.split(' ', 1)
        if len(parts) < 2:
            continue
            
        uttid = parts[0]
        ref_content = parts[1]
        
        # Find corresponding hypothesis
        hyp_line = hyp_dict.get(uttid, '')
        hyp_content = hyp_line.split(' ', 1)[1] if len(hyp_line.split(' ', 1)) > 1 else ''
        
        # Use space-separated tokens as characters (same as ESPnet)
        # Calculate CER and error types
        cer, (correct, sub, del_, ins) = calculate_cer(ref_content, hyp_content)
        
        # Count reference character tokens
        ref_len = len(ref_content.split())
        
        # Update totals
        total_ref_chars += ref_len
        total_correct += correct
        total_sub += sub
        total_del += del_
        total_ins += ins
        
        results.append({
            'uttid': uttid,
            'cer': cer,
            'ref': ref_content,
            'hyp': hyp_content,
            'correct': correct,
            'sub': sub,
            'del': del_,
            'ins': ins,
            'ref_len': ref_len
        })
    
    # Calculate overall CER
    overall_cer = (total_sub + total_del + total_ins) / total_ref_chars if total_ref_chars > 0 else 0
    
    return results, overall_cer, (total_correct, total_sub, total_del, total_ins, total_ref_chars)

def write_results(results, overall_cer, total_stats, output_file):
    """結果をファイルに書き出す (ESPnet形式)"""
    total_correct, total_sub, total_del, total_ins, total_ref_chars = total_stats
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        # 全体の統計情報
        f.write("=" * 80 + "\n")
        f.write("【全体の統計情報】(ESPnet SCLITE形式)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Scores: (#C #S #D #I) {total_correct} {total_sub} {total_del} {total_ins}\n")
        f.write(f"正解文の総文字数（トークン数）：{total_ref_chars}\n")
        f.write(f"文字誤り率（CER）：{overall_cer:.4f} ({overall_cer*100:.2f}%)\n")
        f.write("\n")
        f.write(f"誤りの内訳：\n")
        f.write(f"・正解数（#C）：{total_correct}トークン\n")
        f.write(f"・置換誤り（#S）：{total_sub}トークン ({total_sub/total_ref_chars*100:.2f}%)\n")
        f.write(f"・削除誤り（#D）：{total_del}トークン ({total_del/total_ref_chars*100:.2f}%)\n")
        f.write(f"・挿入誤り（#I）：{total_ins}トークン ({total_ins/total_ref_chars*100:.2f}%)\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 各発話ごとの詳細結果
        f.write("【発話ごとの詳細結果】\n")
        f.write("=" * 80 + "\n\n")
        
        for res in results:
            f.write(f"発話ID：{res['uttid']}\n")
            f.write(f"Scores: (#C #S #D #I) {res['correct']} {res['sub']} {res['del']} {res['ins']}\n")
            f.write(f"文字誤り率（CER）：{res['cer']:.4f} ({res['cer']*100:.2f}%)\n")
            f.write(f"REF: {res['ref']}\n")
            f.write(f"HYP: {res['hyp']}\n")
            f.write("-" * 80 + "\n\n")

def main():
    # ファイルからテキストを読み込む
    try:
        with open('REF/text', 'r', encoding='utf-8') as f:
            ref_lines = f.readlines()
        with open('HYP/text', 'r', encoding='utf-8') as f:
            hyp_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        print("REF/textとHYP/textファイルが必要です。")
        return
    except Exception as e:
        print(f"エラー: {e}")
        return

    # 空行を除去して各行をリストとして保持
    ref_lines = [line.strip() for line in ref_lines if line.strip()]
    hyp_lines = [line.strip() for line in hyp_lines if line.strip()]

    # CERを計算
    results, overall_cer, total_stats = process_recognition_results(ref_lines, hyp_lines)

    # 結果をファイルに書き出し
    output_file = 'result.txt'
    write_results(results, overall_cer, total_stats, output_file)
    
    print(f"解析が完了しました。結果は{output_file}に保存されました。")
    print(f"全体の文字誤り率（CER）：{overall_cer:.4f}")

if __name__ == "__main__":
    main()
    