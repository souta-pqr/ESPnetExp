# Multitask Transducer 実装調査レポート

## 調査日時
2025-10-16

## 調査対象
Multitask RNN-Transducer for ASR + Disfluency Detection

---

## 1. 実装状況の確認

### ✅ 実装完了した機能

#### 1.1 モデルアーキテクチャ
**ファイル:** `/home/kobori/2025/espnet/espnet2/asr/multitask_rnnt_model.py`

✅ **MultitaskRNNTModel**
- 親クラス: `ESPnetASRModel`
- CTC依存: **完全に削除済み**
  - `from espnet2.asr.ctc import CTC` → 削除
  - `ctc` パラメータ → `None`固定
  - `ctc_weight`, `interctc_weight` → サポート対象から除外

```python
# CTC完全削除: ctc=None固定（親クラスとの互換性のため）
ctc = None

supported_kwargs = {
    'ignore_id', 'lsm_weight',
    # 'ctc_weight', 'interctc_weight' は削除（使わない）
    ...
}
```

✅ **Forward Pass**
```python
def forward(self, speech, speech_lengths, text, text_lengths, isdysfl, ...):
    # Extract features
    feats, feats_lengths = self._extract_feats(speech, speech_lengths)
    
    # Encode
    encoder_out, encoder_out_lens = self.encode(feats, feats_lengths)
    
    # Decoder forward pass
    # ✅ CTC不要！Transducerのjoint logitsからアライメントを取得
    loss, loss_dict = self.decoder(
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        labels=text,
        label_lens=text_lengths,
        disfluency_labels=isdysfl,  # ← disfluencyラベルを渡す
    )
```

#### 1.2 Decoder実装
**ファイル:** `/home/kobori/2025/espnet/espnet2/asr/decoder/multitask_transducer.py`

✅ **MultitaskTransducerDecoder**
- 親クラス: `TransducerDecoder`
- Joint Network: `MultitaskJointNetwork`

✅ **Forward Pass (Loss計算)**
```python
def forward(self, encoder_out, encoder_out_lens, labels, label_lens, disfluency_labels):
    # Predict decoder outputs
    decoder_out = self.predict(labels, label_lens)
    
    # Joint network forward
    joint_outputs = self.joint_network(encoder_out, decoder_out)
    asr_scores = joint_outputs["asr_scores"]         # (B, T, U, vocab)
    disfluency_scores = joint_outputs["disfluency_scores"]  # (B, T, U, 4)
    
    # ✅ CTC不要！asr_scoresからTransducerのアライメントを抽出
    loss, loss_dict = self.criterion(
        asr_scores=asr_scores,        # ← Transducerのjoint logits
        targets=labels,
        encoder_lens=encoder_out_lens,
        target_lens=label_lens,
        disfluency_scores=disfluency_scores,
        disfluency_targets=disfluency_labels,
        # ctc_logits 削除！
    )
```

#### 1.3 Joint Network実装
**ファイル:** `/home/kobori/2025/espnet/espnet2/asr/layers/multitask_joint_network.py`

✅ **MultitaskJointNetwork**
- ASR用のJoint Network
- Disfluency用の独立したJoint Network

```python
def forward(self, encoder_out, decoder_out, return_disfluency=True):
    # ASR Joint
    asr_joint = encoder_proj + decoder_proj
    asr_scores = asr_output_layer(asr_joint)  # (B, T, U, vocab)
    
    # Disfluency Joint
    disf_joint = disf_acoustic_proj + disf_decoder_proj
    disfluency_scores = disfluency_output_layer(disf_joint)  # (B, T, U, 4)
    
    return {
        "asr_scores": asr_scores,
        "disfluency_scores": disfluency_scores
    }
```

#### 1.4 Transducer Viterbi Alignment
**ファイル:** `/home/kobori/2025/espnet/espnet2/asr/layers/multitask_joint_network.py`

✅ **transducer_viterbi_alignment()**
- Transducerの2D格子 (T × U) 上でViterbi探索
- **CTCではなくTransducerのlogitsを使用**

```python
def transducer_viterbi_alignment(
    joint_logits: torch.Tensor,  # (B, T, U, vocab) ← Transducer!
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """
    Returns:
        alignments: (B, T) - 各フレームが対応する文字index (-1=blank)
    """
```

✅ **align_disfluency_labels_with_transducer()**
- Transducerアライメントを使ってdisfluencyラベルを対応付け

```python
def align_disfluency_labels_with_transducer(
    joint_logits: torch.Tensor,  # asr_scores from joint network
    text: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    text_lengths: torch.Tensor,
    isdysfl: torch.Tensor,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        frame_level_labels: (B, T) - フレームレベルのdisfluencyラベル
        frame_to_char_idx: (B, T) - 各フレームの文字index
    """
```

#### 1.5 Loss計算
**ファイル:** `/home/kobori/2025/espnet/espnet2/asr/layers/multitask_joint_network.py`

✅ **MultitaskTransducerLoss**
```python
def forward(
    self,
    asr_scores: torch.Tensor,       # (B, T, U, V)
    targets: torch.Tensor,
    encoder_lens: torch.Tensor,
    target_lens: torch.Tensor,
    disfluency_scores: Optional[torch.Tensor] = None,
    disfluency_targets: Optional[torch.Tensor] = None,
    # ✅ ctc_logits削除！
):
    # 1. ASR Loss (RNN-T)
    asr_loss = rnnt_loss(asr_scores, targets, encoder_lens, target_lens)
    
    # 2. Disfluency Loss with Transducer alignment
    if self.use_viterbi_alignment:
        # ✅ Transducerアライメント取得
        frame_level_labels, frame_to_char_idx = align_disfluency_labels_with_transducer(
            joint_logits=asr_scores,  # ← Transducerのlogits!
            text=targets,
            encoder_out_lens=encoder_lens,
            text_lengths=target_lens,
            isdysfl=disfluency_targets,
            blank_id=self.blank_id,
        )
        
        # Alignment-aware selection
        valid_mask = frame_to_char_idx >= 0
        selected_scores = disfluency_scores[batch_idx, frame_idx, char_idx, :]
        selected_labels = frame_level_labels[valid_mask]
        
        disfluency_loss = cross_entropy(selected_scores, selected_labels)
    
    # 3. Total Loss
    total_loss = asr_loss + disfluency_weight * disfluency_loss
    
    return total_loss, loss_dict
```

---

## 2. アーキテクチャの確認

### データフロー (CTC完全削除版)

```
音声入力
  ↓
[Encoder] → encoder_out (B, T, 256)
  ↓
[Decoder] → decoder_out (B, U, 512)
  ↓
[Joint Network]
  ↓
  ├─→ [ASR Head] → asr_logits (B, T, U, vocab)
  │                    ↓
  │               [RNN-T Loss] ← ASR学習
  │                    ↓
  │          [Transducer Viterbi Alignment]  ← Transducerのlogitsから
  │                    ↓
  │          frame_to_char_mapping (B, T)
  │                    ↓
  │                    │
  └─→ [Disfluency Head] → disfluency_logits (B, T, U, 4)
                           ↓
                    [Alignment-Aware Selection]
                           ↓
                    selected_logits (N_valid, 4)
                           ↓
                    [CE Loss] ← Disfluency学習

Total Loss = RNN-T Loss + λ × Disfluency Loss
```

### CTC依存の完全削除

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| Import | `from espnet2.asr.ctc import CTC` | 削除 |
| Model parameter | `ctc: CTC = None` | 削除、`ctc=None`固定 |
| CTC initialization | `ctc = CTC(...)` | 削除 |
| Kwargs support | `'ctc_weight', 'interctc_weight'` | 削除 |
| Alignment source | CTC logits `(B, T, vocab)` | Transducer logits `(B, T, U, vocab)` |
| Alignment function | `viterbi_alignment()` (1D) | `transducer_viterbi_alignment()` (2D) |
| Loss function parameter | `ctc_logits` | 削除 |

---

## 3. 現在のエラー分析

### エラー内容
```
/pytorch/aten/src/ATen/native/cuda/Loss.cu:242: 
nll_loss_forward_reduce_cuda_kernel_2d: 
Assertion `t >= 0 && t < n_classes` failed.

RuntimeError: CUDA error: device-side assert triggered
```

### エラーの原因
**Disfluency損失計算時にラベルインデックスが不正**

```python
# CrossEntropyLossで使用するラベルが範囲外
# 正常範囲: 0 <= label < 4 (disfluency_classes=4)
# エラー: label < 0 または label >= 4
```

### 発生箇所の推定
```python
# MultitaskTransducerLoss.forward()内
disfluency_loss = self.disfluency_criterion(selected_scores, selected_labels)
#                                                             ^^^^^^^^^^^^^^^^
#                                                             ラベルが不正！
```

### 考えられる原因

1. **`align_disfluency_labels_with_transducer()`の問題**
   - `frame_level_labels`に不正な値（-100以外の負数、4以上の値）が含まれる
   - Viterbiアライメントの実装バグ

2. **`char_idx`の範囲外アクセス**
   ```python
   char_idx = frame_to_char_idx[valid_mask]
   selected_scores = disfluency_scores[batch_idx, frame_idx, char_idx, :]
   #                                                          ^^^^^^^^
   # char_idxがUの範囲を超えている可能性
   ```

3. **Disfluencyラベルデータの問題**
   - `isdysfl`に4以上の値や負数が含まれる
   - データ前処理の問題

---

## 4. Transducerとしての正当性

### ✅ 正しく実装されている点

1. **CTC完全削除**
   - CTCへの依存なし
   - Transducerのみで学習

2. **Transducerアライメント使用**
   - Joint logits `(B, T, U, vocab)`から直接アライメント抽出
   - CTCとの矛盾なし

3. **理論的整合性**
   - ASR学習: RNN-T loss
   - Disfluency学習: Transducerのアライメントを使用
   - 一貫したフレームワーク

4. **実装完全性**
   - Model, Decoder, Joint Network, Loss全て実装済み
   - Beam search対応済み

### ❌ 解決すべき問題

1. **ラベルインデックスの検証不足**
   - Disfluencyラベルの範囲チェックが必要
   - アライメント結果の検証が必要

2. **デバッグ情報の不足**
   - エラー発生時の詳細情報なし
   - 中間変数の値が不明

---

## 5. 修正方針

### 優先度1: エラーの特定と修正

1. **ラベル範囲の検証追加**
   ```python
   # align_disfluency_labels_with_transducer()内
   assert torch.all((isdysfl >= 0) & (isdysfl < disfluency_classes))
   ```

2. **アライメント結果の検証**
   ```python
   # char_idxが範囲内か確認
   assert torch.all((char_idx >= 0) & (char_idx < U))
   ```

3. **デバッグログ追加**
   ```python
   print(f"frame_level_labels range: [{frame_level_labels.min()}, {frame_level_labels.max()}]")
   print(f"char_idx range: [{char_idx.min()}, {char_idx.max()}], U={U}")
   print(f"isdysfl range: [{isdysfl.min()}, {isdysfl.max()}]")
   ```

### 優先度2: use_viterbi_alignmentフラグの確認

設定ファイルで`use_viterbi_alignment: true`が設定されているか確認

### 優先度3: データ検証

Disfluencyラベルデータ (`isdysfl`) の値が正しいか確認
- 期待値: 0, 1, 2, 3 (4クラス)
- 不正値: -1, -100, 4以上

---

## 6. まとめ

### 実装状況

✅ **完成している機能:**
- Multitask RNN-Transducer model
- CTC完全削除
- Transducerベースのアライメント抽出
- Alignment-aware disfluency loss

❌ **未解決の問題:**
- Disfluency lossでのラベルインデックスエラー
- 原因: データ検証不足またはアライメント実装のバグ

### Transducerとしての評価

**結論: 理論的に正しいMultitask Transducerが実装されている**

- ✅ CTCへの依存なし
- ✅ Transducerのアライメント使用
- ✅ 一貫したフレームワーク
- ❌ ただし、実行時エラーが存在

### 次のステップ

1. **immediate**: ラベル範囲エラーのデバッグ
2. **short-term**: データ検証とエラーハンドリング追加
3. **medium-term**: 学習実行とメトリクス確認
4. **long-term**: 性能評価と最適化
