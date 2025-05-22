# Improved Disfluency Detection Rate Calculator for Multiple Categories
import re
from typing import List, Tuple, Dict

def calculate_detection_rate(hyp_path, ref_path, output_path, overlap_threshold=0.5):
    """
    バイナリ形式の非流暢性検出結果を評価する
    
    Args:
        hyp_path: 認識結果ファイルのパス
        ref_path: 正解ファイルのパス  
        output_path: 結果出力ファイルのパス
        overlap_threshold: セグメントの一致判定閾値（IoU）
    """
    
    # ファイルを読み込む
    hyp_data = read_dysfl_file(hyp_path)
    ref_data = read_dysfl_file(ref_path)
    
    # 詳細出力ファイル名を生成
    detail_path = output_path.replace('.txt', '_detail.txt')
    
    # 出力ファイルを開く
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 詳細出力用のファイルを開く
        with open(detail_path, 'w', encoding='utf-8') as detail_file:
            
            # セグメントレベルの統計
            total_seg_tp = 0
            total_seg_fp = 0  
            total_seg_fn = 0
            total_ref_segments = 0
            total_hyp_segments = 0
            
            # トークンレベルの統計
            total_tok_tp = 0  # 正しく1と予測
            total_tok_fp = 0  # 間違って1と予測（REFは0）
            total_tok_fn = 0  # 見逃した1（REFは1、HYPは0）
            total_tok_tn = 0  # 正しく0と予測
            
            # 共通の発話IDを取得
            common_ids = set(hyp_data.keys()) & set(ref_data.keys())
            
            if not common_ids:
                out_file.write("エラー: 共通の発話IDが見つかりません\n")
                return
            
            out_file.write(f"評価対象発話数: {len(common_ids)}\n")
            out_file.write(f"IoU閾値: {overlap_threshold}\n\n")
            
            for utt_id in sorted(common_ids):
                hyp_labels = hyp_data[utt_id]
                ref_labels = ref_data[utt_id]
                
                # 長さを合わせる（短い方に合わせる）
                min_len = min(len(hyp_labels), len(ref_labels))
                hyp_labels = hyp_labels[:min_len]
                ref_labels = ref_labels[:min_len]
                
                # トークンレベルのTP/FP/FN/TNを計算
                tok_tp = sum(1 for h, r in zip(hyp_labels, ref_labels) if h == 1 and r == 1)
                tok_fp = sum(1 for h, r in zip(hyp_labels, ref_labels) if h == 1 and r == 0)
                tok_fn = sum(1 for h, r in zip(hyp_labels, ref_labels) if h == 0 and r == 1)
                tok_tn = sum(1 for h, r in zip(hyp_labels, ref_labels) if h == 0 and r == 0)
                
                # セグメントに変換
                hyp_segments = labels_to_segments(hyp_labels)
                ref_segments = labels_to_segments(ref_labels)
                
                # セグメントレベルのアライメントを取得
                seg_tp, seg_fp, seg_fn, alignment = align_segments(ref_segments, hyp_segments, overlap_threshold)
                
                # 統計を更新
                total_seg_tp += seg_tp
                total_seg_fp += seg_fp
                total_seg_fn += seg_fn
                total_ref_segments += len(ref_segments)
                total_hyp_segments += len(hyp_segments)
                
                total_tok_tp += tok_tp
                total_tok_fp += tok_fp
                total_tok_fn += tok_fn
                total_tok_tn += tok_tn
                
                # 発話レベルの結果を計算
                # セグメントレベル
                seg_precision = seg_tp / len(hyp_segments) * 100 if len(hyp_segments) > 0 else 0
                seg_recall = seg_tp / len(ref_segments) * 100 if len(ref_segments) > 0 else 0
                seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if seg_precision + seg_recall > 0 else 0
                
                # トークンレベル
                tok_precision = tok_tp / (tok_tp + tok_fp) * 100 if (tok_tp + tok_fp) > 0 else 0
                tok_recall = tok_tp / (tok_tp + tok_fn) * 100 if (tok_tp + tok_fn) > 0 else 0
                tok_f1 = 2 * tok_precision * tok_recall / (tok_precision + tok_recall) if tok_precision + tok_recall > 0 else 0
                
                # トークン全体の正解率（参考値）
                tok_accuracy = (tok_tp + tok_tn) / len(hyp_labels) * 100 if len(hyp_labels) > 0 else 0
                
                # 結果を出力ファイルに書き込む
                out_file.write(f"発話ID: {utt_id}\n")
                out_file.write(f"REF ラベル: {' '.join(map(str, ref_labels))}\n")
                out_file.write(f"HYP ラベル: {' '.join(map(str, hyp_labels))}\n")
                out_file.write(f"REF セグメント: {ref_segments}\n")
                out_file.write(f"HYP セグメント: {hyp_segments}\n")
                out_file.write(f"セグメント検出率 - Precision: {seg_precision:.2f}%, Recall: {seg_recall:.2f}%, F1: {seg_f1:.2f}%\n")
                out_file.write(f"トークン検出率 - Precision: {tok_precision:.2f}%, Recall: {tok_recall:.2f}%, F1: {tok_f1:.2f}%\n")
                out_file.write(f"トークン正解率: {tok_accuracy:.2f}%\n")
                out_file.write(f"セグメント: TP={seg_tp}, FP={seg_fp}, FN={seg_fn}\n")
                out_file.write(f"トークン: TP={tok_tp}, FP={tok_fp}, FN={tok_fn}, TN={tok_tn}\n")
                out_file.write("\n")
                
                # 詳細出力
                detail_file.write(f"発話ID: {utt_id}\n")
                detail_file.write(f"REF ラベル: {' '.join(map(str, ref_labels))}\n")
                detail_file.write(f"HYP ラベル: {' '.join(map(str, hyp_labels))}\n")
                detail_file.write(f"REF セグメント: {ref_segments}\n")
                detail_file.write(f"HYP セグメント: {hyp_segments}\n")
                
                # トークンレベルの詳細分析（FP/FNのみ表示）
                detail_file.write("トークンレベル誤り分析:\n")
                error_found = False
                for i, (h, r) in enumerate(zip(hyp_labels, ref_labels)):
                    if h == 1 and r == 0:
                        detail_file.write(f"  位置{i}: FP (0->1)\n")
                        error_found = True
                    elif h == 0 and r == 1:
                        detail_file.write(f"  位置{i}: FN (1->0)\n")
                        error_found = True
                if not error_found:
                    detail_file.write("  誤りなし\n")
                
                detail_file.write("セグメントアライメント:\n")
                for ref_seg, hyp_seg, match_type in alignment:
                    if match_type == "TP":
                        detail_file.write(f"  {ref_seg} -- {hyp_seg} (一致, IoU={calculate_iou(ref_seg, hyp_seg):.3f})\n")
                    elif match_type == "FN":
                        detail_file.write(f"  {ref_seg} -- (脱落)\n")
                    elif match_type == "FP":
                        detail_file.write(f"  -- {hyp_seg} (挿入)\n")
                
                detail_file.write("\n")
            
            # 全体の結果を計算
            # セグメントレベル
            total_seg_precision = total_seg_tp / total_hyp_segments * 100 if total_hyp_segments > 0 else 0
            total_seg_recall = total_seg_tp / total_ref_segments * 100 if total_ref_segments > 0 else 0
            total_seg_f1 = 2 * total_seg_precision * total_seg_recall / (total_seg_precision + total_seg_recall) if total_seg_precision + total_seg_recall > 0 else 0
            
            # トークンレベル
            total_tok_precision = total_tok_tp / (total_tok_tp + total_tok_fp) * 100 if (total_tok_tp + total_tok_fp) > 0 else 0
            total_tok_recall = total_tok_tp / (total_tok_tp + total_tok_fn) * 100 if (total_tok_tp + total_tok_fn) > 0 else 0
            total_tok_f1 = 2 * total_tok_precision * total_tok_recall / (total_tok_precision + total_tok_recall) if total_tok_precision + total_tok_recall > 0 else 0
            
            # トークン全体の正解率
            total_tokens = total_tok_tp + total_tok_fp + total_tok_fn + total_tok_tn
            total_tok_accuracy = (total_tok_tp + total_tok_tn) / total_tokens * 100 if total_tokens > 0 else 0
            
            # 全体の結果を出力
            out_file.write("=" * 60 + "\n")
            out_file.write("全体での検出率:\n")
            out_file.write("=" * 60 + "\n")
            out_file.write("【セグメントレベル統計】\n")
            out_file.write(f"総REFセグメント数: {total_ref_segments}\n")
            out_file.write(f"総HYPセグメント数: {total_hyp_segments}\n")
            out_file.write(f"セグメント TP: {total_seg_tp}, FP: {total_seg_fp}, FN: {total_seg_fn}\n")
            out_file.write("\n")
            out_file.write("【トークンレベル統計】\n")
            out_file.write(f"総トークン数: {total_tokens}\n")
            out_file.write(f"REF中の1の数: {total_tok_tp + total_tok_fn}\n")
            out_file.write(f"HYP中の1の数: {total_tok_tp + total_tok_fp}\n")
            out_file.write(f"トークン TP: {total_tok_tp}, FP: {total_tok_fp}, FN: {total_tok_fn}, TN: {total_tok_tn}\n")
            out_file.write("\n")
            out_file.write("【セグメントレベル評価】\n")
            out_file.write(f"Precision: {total_seg_precision:.2f}%\n")
            out_file.write(f"Recall: {total_seg_recall:.2f}%\n")
            out_file.write(f"F1-score: {total_seg_f1:.2f}%\n")
            out_file.write("\n")
            out_file.write("【トークンレベル評価】\n")
            out_file.write(f"Precision: {total_tok_precision:.2f}%\n")
            out_file.write(f"Recall: {total_tok_recall:.2f}%\n")
            out_file.write(f"F1-score: {total_tok_f1:.2f}%\n")
            out_file.write(f"Accuracy: {total_tok_accuracy:.2f}%\n")


def read_dysfl_file(file_path: str) -> Dict[str, List[int]]:
    """
    非流暢性ファイルを読み込む
    
    Args:
        file_path: ファイルパス
        
    Returns:
        Dict[発話ID, ラベルリスト]
    """
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Line {line_num} in {file_path} has insufficient data: {line}")
                    continue
                    
                utt_id = parts[0]
                try:
                    labels = [int(x) for x in parts[1:]]
                    data[utt_id] = labels
                except ValueError as e:
                    print(f"Warning: Line {line_num} in {file_path} has invalid label: {line}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    
    return data


def labels_to_segments(labels: List[int]) -> List[Tuple[int, int]]:
    """
    バイナリラベルをセグメント（開始位置、終了位置）のリストに変換
    
    Args:
        labels: バイナリラベルのリスト
        
    Returns:
        セグメントのリスト [(start, end), ...]
    """
    segments = []
    start = None
    
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            # セグメントの開始
            start = i
        elif label == 0 and start is not None:
            # セグメントの終了
            segments.append((start, i - 1))
            start = None
    
    # 最後まで1が続いている場合
    if start is not None:
        segments.append((start, len(labels) - 1))
    
    return segments


def calculate_iou(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
    """
    2つのセグメントのIoU（Intersection over Union）を計算
    
    Args:
        seg1: セグメント1 (start, end)
        seg2: セグメント2 (start, end)
        
    Returns:
        IoU値（0.0〜1.0）
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    # 交集合の計算
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    
    # 和集合の計算
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    
    return intersection / union if union > 0 else 0.0


def align_segments(ref_segments: List[Tuple[int, int]], 
                  hyp_segments: List[Tuple[int, int]], 
                  threshold: float = 0.5) -> Tuple[int, int, int, List]:
    """
    セグメント間のアライメントを取得し、TP/FP/FNを計算
    
    Args:
        ref_segments: 正解セグメント
        hyp_segments: 認識セグメント
        threshold: IoUの閾値
        
    Returns:
        (TP, FP, FN, alignment)
    """
    ref_matched = [False] * len(ref_segments)
    hyp_matched = [False] * len(hyp_segments)
    alignment = []
    
    # 各HYPセグメントに対して最適なREFセグメントを探す
    for h_idx, hyp_seg in enumerate(hyp_segments):
        best_iou = 0
        best_ref_idx = -1
        
        for r_idx, ref_seg in enumerate(ref_segments):
            if ref_matched[r_idx]:
                continue
                
            iou = calculate_iou(ref_seg, hyp_seg)
            if iou > best_iou and iou >= threshold:
                best_iou = iou
                best_ref_idx = r_idx
        
        if best_ref_idx >= 0:
            # マッチング成功
            ref_matched[best_ref_idx] = True
            hyp_matched[h_idx] = True
            alignment.append((ref_segments[best_ref_idx], hyp_seg, "TP"))
    
    # マッチしなかったREFセグメント（FN）
    for r_idx, ref_seg in enumerate(ref_segments):
        if not ref_matched[r_idx]:
            alignment.append((ref_seg, None, "FN"))
    
    # マッチしなかったHYPセグメント（FP）
    for h_idx, hyp_seg in enumerate(hyp_segments):
        if not hyp_matched[h_idx]:
            alignment.append((None, hyp_seg, "FP"))
    
    tp = sum(ref_matched)
    fn = len(ref_segments) - tp
    fp = len(hyp_segments) - sum(hyp_matched)
    
    return tp, fp, fn, alignment


def evaluate_all_categories():
    """
    全カテゴリの評価を実行し、結果をまとめる
    """
    categories = ["filler", "disfluency", "interjection"]
    results_summary = {}
    
    print("=" * 60)
    print("多カテゴリ非流暢性検出評価")
    print("=" * 60)
    
    for category in categories:
        print(f"\n{category.capitalize()}カテゴリを評価中...")
        hyp_file = f"HYP/isdysfl_{category}"
        ref_file = f"REF/isdysfl_{category}"
        output_file = f"{category}_detection_rate.txt"
        
        try:
            calculate_detection_rate(hyp_file, ref_file, output_file, overlap_threshold=0.5)
            print(f"✓ 結果は {output_file} に保存されました")
            print(f"✓ 詳細結果は {category}_detection_rate_detail.txt に保存されました")
            
            # 結果の要約を抽出
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "【セグメントレベル評価】" in content:
                        seg_start = content.find("【セグメントレベル評価】")
                        tok_start = content.find("【トークンレベル評価】")
                        if seg_start != -1 and tok_start != -1:
                            seg_section = content[seg_start:tok_start]
                            tok_section = content[tok_start:]
                            
                            # 数値を抽出（簡易的）
                            import re
                            seg_f1_match = re.search(r'F1-score: ([\d.]+)%', seg_section)
                            tok_f1_match = re.search(r'F1-score: ([\d.]+)%', tok_section)
                            
                            if seg_f1_match and tok_f1_match:
                                results_summary[category] = {
                                    'segment_f1': float(seg_f1_match.group(1)),
                                    'token_f1': float(tok_f1_match.group(1))
                                }
            except:
                pass
                
        except FileNotFoundError as e:
            print(f"✗ ファイルが見つかりません: {e}")
        except Exception as e:
            print(f"✗ エラーが発生しました: {e}")
    
    # 全体のサマリーを出力
    if results_summary:
        print("\n" + "=" * 60)
        print("評価結果サマリー")
        print("=" * 60)
        print(f"{'カテゴリ':<15} {'セグメント F1':<15} {'トークン F1':<15}")
        print("-" * 50)
        
        for category, scores in results_summary.items():
            print(f"{category:<15} {scores['segment_f1']:<15.2f} {scores['token_f1']:<15.2f}")
        
        # 平均値
        avg_seg_f1 = sum(scores['segment_f1'] for scores in results_summary.values()) / len(results_summary)
        avg_tok_f1 = sum(scores['token_f1'] for scores in results_summary.values()) / len(results_summary)
        print("-" * 50)
        print(f"{'平均':<15} {avg_seg_f1:<15.2f} {avg_tok_f1:<15.2f}")


# 使用例
if __name__ == "__main__":
    # 全カテゴリの評価を実行
    evaluate_all_categories()