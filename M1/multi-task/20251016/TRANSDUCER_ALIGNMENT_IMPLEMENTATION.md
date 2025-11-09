# Transducerアライメント実装 - CTC依存削除

## 問題点
従来の実装ではCTCのViterbiアライメントを使用してdisfluency検出のアライメントを取得していた。
これはTransducerアーキテクチャと矛盾する：
- RNN-Tは独自のアライメントをforward-backwardで学習
- CTCとRNN-Tは異なるアライメントを学習
- CTCに依存すると純粋なTransducer学習にならない

## 解決方法
**Transducerのjoint network出力から直接Viterbiアライメントを抽出**

### 実装の流れ

#### 1. `transducer_viterbi_alignment()` 関数（新規）
```python
def transducer_viterbi_alignment(
    joint_logits: torch.Tensor,  # (B, T, U, vocab_size)
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
```

**アルゴリズム：**
- Transducerの格子（lattice）上でViterbi探索
- 状態 (t, u): 時刻tで、u個のトークンを出力済み
- 遷移:
  - Blank遷移: (t, u) → (t+1, u) - 時間進む、トークン変わらず
  - Label遷移: (t, u) → (t, u+1) - targets[u]を出力

**CTCとの違い：**
- CTC: フレームごとに独立した予測
- Transducer: (時間, トークン位置)の2次元格子で予測

#### 2. `align_disfluency_labels_with_transducer()` 関数
```python
def align_disfluency_labels_with_transducer(
    joint_logits: torch.Tensor,  # Transducerのjoint output
    encoder_out_lens: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    isdysfl: torch.Tensor,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**変更点：**
- `ctc_logits`パラメータを削除
- `joint_logits`（Transducerのjoint network出力）を使用
- `transducer_viterbi_alignment()`を呼び出し

#### 3. Loss計算の変更
`MultitaskTransducerLoss.forward()`

```python
def forward(
    self,
    asr_scores: torch.Tensor,  # これをアライメント抽出に使用！
    targets: torch.Tensor,
    encoder_lens: torch.Tensor,
    target_lens: torch.Tensor,
    disfluency_scores: Optional[torch.Tensor] = None,
    disfluency_targets: Optional[torch.Tensor] = None,
    # ctc_logits削除！
    # encoder_out削除！
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```

**重要な変更：**
```python
# 旧: CTCからアライメント
frame_level_labels, frame_to_char_idx = align_disfluency_labels_with_viterbi(
    ctc_logits=ctc_logits,  # ❌ CTC依存
    ...
)

# 新: Transducerからアライメント
frame_level_labels, frame_to_char_idx = align_disfluency_labels_with_transducer(
    joint_logits=asr_scores,  # ✅ Transducerのjoint logits
    ...
)
```

#### 4. モデル側の変更
`MultitaskRNNTModel.forward()` と `MultitaskTransducerDecoder.forward()`

```python
# 旧: CTCロジット計算
ctc_logits = None
if self.ctc is not None and isdysfl is not None:
    ctc_logits = self.ctc.ctc_lo(encoder_out)  # ❌

loss, loss_dict = self.decoder(
    ...,
    ctc_logits=ctc_logits,  # ❌
    encoder_out=encoder_out,  # ❌
)

# 新: CTC不要
loss, loss_dict = self.decoder(
    encoder_out=encoder_out,
    encoder_out_lens=encoder_out_lens,
    labels=text,
    label_lens=text_lengths,
    disfluency_labels=isdysfl,
    # ctc_logits削除！✅
)
```

## 利点

### 1. 理論的に正しい
- TransducerはTransducerのアライメントを使用
- CTCとの矛盾がない
- 純粋なTransducer学習

### 2. シンプル
- CTC計算が不要
- 依存関係が減少
- コードが整理される

### 3. 一貫性
- ASRとdisfluency検出が同じアライメントを使用
- Transducerのforward-backward学習と整合

## アーキテクチャの流れ

```
音声入力
  ↓
Encoder (Conformer)
  ↓
  ├─→ Joint Network (ASR) → asr_scores (B, T, U, vocab_size)
  │     ↑                      ↓
  │     └── Decoder ──────────┤
  │                            ├─→ RNN-T Loss (ASR)
  │                            │
  │                            └─→ Viterbi Alignment抽出
  │                                 (Transducerの格子から)
  │                                       ↓
  └─→ Joint Network (Disf) → disfluency_scores (B, T, U, 4)
        ↑                            ↓
        └── Decoder ─────────────────┤
                                     │
                            Alignment-Aware Selection
                            (frame_to_char_idxで選択)
                                     ↓
                            Cross-Entropy Loss (Disfluency)
```

## 実装ファイル

### 変更されたファイル：
1. `/home/kobori/2025/espnet/espnet2/asr/layers/multitask_joint_network.py`
   - `transducer_viterbi_alignment()` 追加（新規関数）
   - `align_disfluency_labels_with_transducer()` 追加（旧関数のTransducer版）
   - `MultitaskTransducerLoss.forward()` 更新（CTC削除）

2. `/home/kobori/2025/espnet/espnet2/asr/multitask_rnnt_model.py`
   - CTC logits計算を削除
   - `ctc_logits`パラメータを削除

3. `/home/kobori/2025/espnet/espnet2/asr/decoder/multitask_transducer.py`
   - `forward()`から`ctc_logits`パラメータを削除

## 次のステップ

1. **テスト実行**
   ```bash
   cd /home/kobori/2025/espnet/egs2/cejc/asr1
   # 既存のトレーニングを再開または新規開始
   ```

2. **エラーログ確認**
   - Transducer Viterbiアライメントが正しく動作するか
   - loss計算が成功するか
   - `t >= 0 && t < n_classes`エラーが解消されるか

3. **デバッグポイント**
   - `transducer_viterbi_alignment()`の出力確認
   - `frame_to_char_idx`の値範囲確認（0 ≤ idx < L または -1）
   - `selected_scores`と`selected_labels`の形状確認

## 理論的背景

### Transducer Lattice
```
u=0  u=1  u=2  ... u=L
 ●────●────●─...──●  t=0
 │╲   │╲   │╲     │
 │ ╲  │ ╲  │ ╲    │
 │  ╲ │  ╲ │  ╲   │
 ●────●────●─...──●  t=1
 │╲   │╲   │╲     │
 ⋮    ⋮    ⋮      ⋮
 ●────●────●─...──●  t=T-1

横方向（→）: Label遷移（トークン出力）
縦方向（↓）: Blank遷移（時間進行）
```

### Viterbi探索
各状態(t, u)での最大確率を計算：
- `viterbi[t, u]` = 状態(t, u)に到達する最大対数確率
- Blank遷移: `viterbi[t, u] += log P(blank | t-1, u)`
- Label遷移: `viterbi[t, u] += log P(label[u] | t, u-1)`

最終状態(T-1, L)から逆トレースして最適パスを取得。

### アライメント抽出
各時刻tでのトークン位置uを記録：
- `alignments[b, t] = u` → フレームtはトークンuに対応
- `alignments[b, t] = -1` → フレームtはblank

これを使ってdisfluency labelをフレームレベルに展開。
