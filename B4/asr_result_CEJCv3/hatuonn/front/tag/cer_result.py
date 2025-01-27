import numpy as np
from datetime import datetime

def calculate_cer(ref, hyp):
    """
    Calculate Character Error Rate between reference (correct) and hypothesis (ASR result) strings
    CER = (S + D + I) / N, where:
    S = number of substitutions
    D = number of deletions
    I = number of insertions
    N = number of characters in reference
    """
    # Create matrix
    dp = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    # Initialize first row and column
    for i in range(len(ref) + 1):
        dp[i][0] = i  # deletions
    for j in range(len(hyp) + 1):
        dp[0][j] = j  # insertions
        
    # Initialize backtrace matrix to track operations
    ops = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    ops[:, 0] = 1  # deletion
    ops[0, :] = 2  # insertion
    
    # Fill the matrix
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
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
    i, j = len(ref), len(hyp)
    substitutions = deletions = insertions = 0
    
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
            i -= 1
            j -= 1
    
    # Calculate CER
    ref_length = len(ref)
    if ref_length == 0:
        return 0, (0, 0, 0)
    
    cer = (substitutions + deletions + insertions) / ref_length
    return cer, (substitutions, deletions, insertions)

def process_recognition_results(ref_lines, hyp_lines):
    """
    Process the recognition results and calculate CER for each utterance
    ref_lines: list of lines from REF/text
    hyp_lines: list of lines from HYP/text
    """
    results = []
    total_ref_chars = 0
    total_sub = total_del = total_ins = 0
    
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
        
        # Remove spaces between characters
        ref_chars = ''.join(ref_content.split())
        hyp_chars = ''.join(hyp_content.split())
        
        # Calculate CER and error types
        cer, (sub, del_, ins) = calculate_cer(ref_chars, hyp_chars)
        
        # Update totals
        total_ref_chars += len(ref_chars)
        total_sub += sub
        total_del += del_
        total_ins += ins
        
        results.append({
            'uttid': uttid,
            'cer': cer,
            'ref': ref_chars,
            'hyp': hyp_chars,
            'sub': sub,
            'del': del_,
            'ins': ins,
            'ref_len': len(ref_chars)
        })
    
    # Calculate overall CER
    overall_cer = (total_sub + total_del + total_ins) / total_ref_chars if total_ref_chars > 0 else 0
    
    return results, overall_cer, (total_sub, total_del, total_ins, total_ref_chars)

def write_results(results, overall_cer, total_stats, output_file):
    """結果をファイルに書き出す"""
    total_sub, total_del, total_ins, total_ref_chars = total_stats
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        # 全体の統計情報
        f.write("【全体の統計情報】\n")
        f.write("-" * 40 + "\n")
        f.write(f"全体の文字誤り率（CER）：{overall_cer:.4f}\n")
        f.write(f"正解文の総文字数：{total_ref_chars}文字\n")
        f.write(f"置換誤り総数：{total_sub}箇所\n")
        f.write(f"削除誤り総数：{total_del}箇所\n")
        f.write(f"挿入誤り総数：{total_ins}箇所\n")
        f.write("\n")
        f.write(f"各種誤りの内訳：\n")
        if total_ref_chars > 0:
            f.write(f"・置換誤り率：{total_sub/total_ref_chars:.4f}\n")
            f.write(f"・削除誤り率：{total_del/total_ref_chars:.4f}\n")
            f.write(f"・挿入誤り率：{total_ins/total_ref_chars:.4f}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 各発話ごとの詳細結果
        f.write("【発話ごとの詳細結果】\n")
        f.write("=" * 80 + "\n\n")
        
        for res in results:
            f.write(f"発話ID：{res['uttid']}\n")
            f.write(f"文字誤り率（CER）：{res['cer']:.4f}\n")
            f.write(f"誤り内訳：\n")
            f.write(f"・置換誤り：{res['sub']}箇所\n")
            f.write(f"・削除誤り：{res['del']}箇所\n")
            f.write(f"・挿入誤り：{res['ins']}箇所\n")
            f.write(f"正解文：{res['ref']}\n")
            f.write(f"認識結果：{res['hyp']}\n")
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