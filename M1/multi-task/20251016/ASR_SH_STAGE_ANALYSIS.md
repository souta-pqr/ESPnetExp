# asr.sh Stage 10以降の確認レポート

## 📊 現在の状況

**Stage 11まで進行中:** ✅ 問題なし
- モデル初期化成功
- Multitask Transducerが正しく動作開始

---

## Stage 10: ASR Collect Stats（統計収集）

### ✅ 正しく実装されている点

1. **Disfluency Detection対応**
```bash
# Disfluency data の追加
if "${use_disfluency_detection}"; then
    if [ -f "${_asr_train_dir}/isdysfl" ] && [ -f "${_asr_valid_dir}/isdysfl" ]; then
        _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/isdysfl,isdysfl,text "
        _opts+="--valid_data_path_and_name_and_type ${_asr_valid_dir}/isdysfl,isdysfl,text "
    fi
fi
```

2. **Disfluency固有のオプション**
```bash
_opts+="--disfluency_weight ${disfluency_weight} "
_opts+="--disfluency_classes ${disfluency_classes} "
_opts+="--report_disfluency_accuracy ${report_disfluency_accuracy} "
_opts+="--use_disfluency_detection ${use_disfluency_detection} "
```

3. **正しいタスクバイナリ使用**
```bash
${python} -m espnet2.bin.${disfluency_task}_train \  # disfluency_asr_train
    --collect_stats true \
    ...
```

4. **Shape file処理**
```bash
# Disfluency shape files
if "${use_disfluency_detection}"; then
    <"${asr_stats_dir}/train/isdysfl_shape" \
        awk -v N="${disfluency_classes}" '{ print $0 "," N }' \
        >"${asr_stats_dir}/train/isdysfl_shape.disfluency"
fi
```

### ❓ 確認すべき点

**なし - Stage 10は完璧に実装されています**

---

## Stage 11: ASR Training（学習）

### ✅ 正しく実装されている点

1. **Multitask Transducer明示的指定**
```bash
if "${use_multitask_transducer}"; then
    _opts+="--decoder multitask_transducer "
    _opts+="--model multitask_rnnt "
fi
```

2. **Disfluencyデータの追加（num_splits_asr > 1の場合）**
```bash
if "${use_disfluency_detection}" && [ -f "${_asr_train_dir}/isdysfl" ]; then
    _split_scps+=" ${_asr_train_dir}/isdysfl ${asr_stats_dir}/train/isdysfl_shape.disfluency"
fi
```

3. **Disfluencyデータの追加（通常の場合）**
```bash
if "${use_disfluency_detection}"; then
    if [ -f "${_asr_train_dir}/isdysfl" ]; then
        _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/isdysfl,isdysfl,text "
        _opts+="--train_shape_file ${asr_stats_dir}/train/isdysfl_shape.disfluency "
    fi
fi
```

4. **Validationデータの追加**
```bash
if "${use_disfluency_detection}"; then
    if [ -f "${_asr_valid_dir}/isdysfl" ]; then
        _opts+="--valid_data_path_and_name_and_type ${_asr_valid_dir}/isdysfl,isdysfl,text "
        _opts+="--valid_shape_file ${asr_stats_dir}/valid/isdysfl_shape.disfluency "
    fi
fi
```

5. **Disfluency固有オプション**
```bash
if "${use_disfluency_detection}"; then
    _opts+="--disfluency_weight ${disfluency_weight} "
    _opts+="--disfluency_classes ${disfluency_classes} "
    _opts+="--report_disfluency_accuracy ${report_disfluency_accuracy} "
    _opts+="--use_disfluency_detection ${use_disfluency_detection} "
fi
```

6. **正しいタスクバイナリ使用**
```bash
${python} -m espnet2.bin.${disfluency_task}_train \  # disfluency_asr_train
    --use_preprocessor true \
    ...
```

### ❓ 確認すべき点

**なし - Stage 11も完璧に実装されています**

---

## Stage 12: Decoding（推論）

### ⚠️ 潜在的な問題点

Stage 12以降のdecoding/evaluationスクリプトは**標準的なASR用**のままです。
Disfluency検出の出力・評価が含まれていない可能性があります。

#### 懸念点

1. **Disfluency予測の出力がされるか？**
   - 標準のASR decodingスクリプトはASRのテキストのみ出力
   - Disfluencyラベルの予測が保存されない可能性

2. **Disfluency評価メトリクスの計算**
   - ASR: CER/WER
   - Disfluency: Accuracy, Precision, Recall, F1-score
   - 評価スクリプトが対応しているか不明

3. **Beam search compatibility**
   - `BeamSearchJointWrapper`はASRのみを返す
   - Disfluency予測をどう取得するか

### 📋 Stage 12の現状確認が必要

**確認すべき内容:**
```bash
# Stage 12 のコードを確認
grep -A 100 "stage.*12.*stop_stage.*12" asr.sh

# Decodingで以下が対応しているか:
# 1. Disfluency予測の出力
# 2. Disfluency評価
# 3. 結果の保存形式
```

---

## Stage 13: Scoring（評価）

### ✅ Disfluency評価スクリプトが含まれている！

```python
#!/usr/bin/env python3
"""
Comprehensive disfluency detection evaluation script.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""

def compute_metrics(ref_labels, hyp_labels, num_classes=4):
    # Accuracy, Precision, Recall, F1-score計算
    ...

def evaluate_disfluency(ref_file, hyp_file, output_file):
    # Confusion matrix作成
    ...
```

**これは良い！** 評価スクリプトは既に実装されています。

### ❓ 確認すべき点

1. **このスクリプトが実際に呼ばれるか？**
   - Stage 13でこのスクリプトが実行される部分を確認

2. **入力ファイルの形式**
   - `ref_file`: 正解ラベル（isdysfl）
   - `hyp_file`: 予測ラベル（decodingで出力されるか？）

---

## 🎯 推奨事項

### 優先度1: Stage 12の詳細確認

Stage 12でdisfluency予測が正しく出力されるか確認が必要です：

```bash
# asr.shのStage 12部分を表示
sed -n '/stage.*12.*stop_stage.*12/,/^fi$/p' asr.sh > stage12_check.txt
```

**確認すべきポイント:**
1. ✅ ASR decodingコマンド
2. ❓ Disfluency予測の出力処理
3. ❓ 出力ファイルの保存先

### 優先度2: 推論時のDisfluency取得方法

**現在の疑問:**
- Beam searchでASRテキストは取得できる
- **Disfluencyラベルはどう取得？**

**解決策の候補:**
1. **方法A**: Beam search後にalignmentを使って予測
2. **方法B**: Forward passで両方を同時に取得
3. **方法C**: 2段階処理（ASR → Disfluency）

### 優先度3: 評価フローの確認

```
学習 (Stage 11)
  ↓
Decoding (Stage 12)  ← ここでdisfluency予測を出力
  ↓
Scoring (Stage 13)   ← disfluency評価スクリプト実行
```

---

## ✅ 結論

### Stage 10-11: 問題なし ✅

- ✅ Disfluency detection完全対応
- ✅ Multitask Transducer正しく設定
- ✅ データ・オプション全て適切
- ✅ 学習は正常に開始

### Stage 12: 要確認 ⚠️

Decodingでdisfluency予測の出力・保存が実装されているか不明。
通常のASR decodingスクリプトのままの可能性。

### Stage 13: 評価スクリプトあり ✅

Disfluency評価用のPythonスクリプトが実装済み。
ただし、Stage 12でdisfluency予測が出力されることが前提。

---

## 📝 次のアクション

1. **Stage 11の学習を継続**
   - 現在問題なし、学習を進める

2. **学習完了後、Stage 12を詳細確認**
   - Disfluency予測の出力方法
   - 必要に応じてスクリプト修正

3. **Decoding実装の選択肢**
   - 既存のASR decodingを拡張
   - または専用のmultitask decoding実装

現時点では**Stage 10-11は完璧に実装されており、問題なく学習できています**。
