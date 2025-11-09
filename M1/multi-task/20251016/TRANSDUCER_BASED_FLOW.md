# Transducerベースのマルチタスク学習フロー（CTC依存削除版）

## 更新後のデータフロー（CTC完全削除版）

```python
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
  │          **[Transducer Viterbi Alignment]**  ← ここが重要！
  │                    ↓
  │          frame_to_char_mapping (B, T)
  │                    ↓ (どのフレームがどの文字に対応するか)
  │                    │
  └─→ [Disfluency Head] → disfluency_logits (B, T, U, 4)
                           ↓
                    [Alignment-Aware Selection]
                           ↓ (frame_to_char_mappingを使用)
                    selected_logits (N_valid, 4)
                           ↓
                    [CE Loss] ← Disfluency学習

Total Loss = RNN-T Loss + λ × Disfluency Loss
(CTC Loss は完全に削除)
```

## 主要な変更点

### 1. アライメント取得元の変更

**旧実装（CTC依存）:**
```python
# CTCのlogitsからViterbiアライメント
ctc_logits (B, T, vocab)
    ↓
viterbi_alignment(ctc_logits, ...)
    ↓
frame_to_char_idx (B, T)
```

**新実装（Transducer純粋）:**
```python
# TransducerのJoint logitsからViterbiアライメント
asr_logits (B, T, U, vocab)  ← Joint Networkの出力
    ↓
transducer_viterbi_alignment(asr_logits, ...)
    ↓
frame_to_char_idx (B, T)
```

### 2. 関数の変更

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| アライメント関数 | `viterbi_alignment()` (CTC用) | `transducer_viterbi_alignment()` (Transducer用) |
| ラベル対応付け | `align_disfluency_labels_with_viterbi()` | `align_disfluency_labels_with_transducer()` |
| 入力データ | `ctc_logits` (B, T, vocab) | `asr_scores` (B, T, U, vocab) |
| アルゴリズム | CTC Viterbi (1D格子) | Transducer Viterbi (2D格子) |

### 3. Loss計算フローの詳細

```python
# 1. Forward pass
encoder_out = encoder(speech)                    # (B, T, 256)
decoder_out = decoder(prev_tokens)               # (B, U, 512)

joint_outputs = joint_network(encoder_out, decoder_out)
asr_scores = joint_outputs["asr_scores"]         # (B, T, U, vocab)
disfluency_scores = joint_outputs["disfluency_scores"]  # (B, T, U, 4)

# 2. ASR Loss (RNN-T)
asr_loss = rnnt_loss(asr_scores, targets, ...)

# 3. Transducerアライメント取得（ここが新しい！）
alignments = transducer_viterbi_alignment(
    asr_scores,          # ← Transducerの出力を直接使用
    targets,
    encoder_lens,
    target_lens,
    blank_id
)
# alignments: (B, T) どのフレームがどの文字に対応するか (-1=blank)

# 4. Character-levelラベルをFrame-levelに変換
frame_level_labels = torch.full((B, T), -100, ...)
for b in range(B):
    for t in range(T):
        char_idx = alignments[b, t]
        if char_idx >= 0:  # non-blank
            frame_level_labels[b, t] = isdysfl[b, char_idx]

# 5. Alignment-Aware Disfluency Loss
valid_mask = (alignments >= 0)  # blanks以外
batch_idx = ...  # valid positions
frame_idx = ...
char_idx = alignments[valid_mask]

# 対応する文字位置のlogitsのみを選択
selected_logits = disfluency_scores[batch_idx, frame_idx, char_idx, :]
selected_labels = frame_level_labels[valid_mask]

disfluency_loss = cross_entropy(selected_logits, selected_labels)

# 6. Total Loss
total_loss = asr_loss + disfluency_weight * disfluency_loss
```

## Transducer Viterbiアルゴリズムの詳細

### CTC vs Transducer Viterbi

**CTC Viterbi (1次元動的計画法):**
```python
# 状態: 各時刻tでどの文字まで出力したか
dp[t][s] = max probability of emitting s symbols by time t

# 遷移:
# - Stay in same state (emit blank)
# - Move to next state (emit character)
```

**Transducer Viterbi (2次元動的計画法):**
```python
# 状態: 時刻tで文字uまで出力した時の確率
dp[t][u] = max probability of being at (t, u)

# 遷移:
# - (t-1, u) → (t, u): フレーム進み (blank emission)
# - (t, u-1) → (t, u): 文字進み (character emission)
# - (t-1, u-1) → (t, u): 両方進み (直接遷移)

# log-domain:
dp[t][u] = max(
    dp[t-1][u] + log P(blank | t, u),      # フレーム進み
    dp[t][u-1] + log P(char_u | t, u),     # 文字進み
)
```

### アライメント抽出

Backtrackingで最適パスを復元：
```python
# 終端: (T-1, U-1)から開始
path = [(T-1, U-1)]

# 逆方向に辿る
for each step backwards:
    if came from (t-1, u):  # blank emission
        frame t is aligned to NO character (mark as -1)
    if came from (t, u-1):  # character emission
        # 複数のフレームが同じ文字に対応する可能性
    if came from (t-1, u-1):  # diagonal
        frame t is aligned to character u

# 結果: frame_to_char[t] = which character index (or -1 for blank)
```

## 実装コードの対応関係

### multitask_joint_network.py

```python
def transducer_viterbi_alignment(
    joint_logits: torch.Tensor,  # (B, T, U, vocab_size)
    targets: torch.Tensor,       # (B, L)
    input_lengths: torch.Tensor, # (B,)
    target_lengths: torch.Tensor,# (B,)
    blank_id: int = 0,
) -> torch.Tensor:
    """
    Transducerの2D格子からViterbiアライメントを計算
    
    Returns:
        alignments: (B, T) - 各フレームが対応する文字のindex (-1=blank)
    """
    # 実装済み
```

```python
def align_disfluency_labels_with_transducer(
    asr_scores: torch.Tensor,        # (B, T, U, vocab)
    text: torch.Tensor,              # (B, L)
    encoder_out_lens: torch.Tensor,  # (B,)
    text_lengths: torch.Tensor,      # (B,)
    isdysfl: torch.Tensor,          # (B, L)
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transducerアライメントを使ってdisfluencyラベルを対応付け
    
    Returns:
        frame_level_labels: (B, T) - フレームレベルのラベル
        frame_to_char_idx: (B, T) - 各フレームの文字index
    """
    # 実装済み
```

### multitask_transducer_loss.py (MultitaskTransducerLoss)

```python
def forward(
    self,
    asr_scores: torch.Tensor,           # (B, T, U, V)
    targets: torch.Tensor,              # (B, L)
    encoder_lens: torch.Tensor,         # (B,)
    target_lens: torch.Tensor,          # (B,)
    disfluency_scores: Optional[torch.Tensor] = None,  # (B, T, U, C)
    disfluency_targets: Optional[torch.Tensor] = None, # (B, L)
    # ctc_logits削除！
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    変更点:
    - ctc_logitsパラメータ削除
    - asr_scoresから直接アライメント取得
    """
    
    # ASR Loss
    asr_loss = rnnt_loss(asr_scores, ...)
    
    # Disfluency Loss with Transducer alignment
    if self.use_disfluency_detection:
        frame_level_labels, frame_to_char_idx = align_disfluency_labels_with_transducer(
            asr_scores=asr_scores,  # ← CTCではなくTransducerのlogits
            text=targets,
            encoder_out_lens=encoder_lens,
            text_lengths=target_lens,
            isdysfl=disfluency_targets,
            blank_id=self.blank_id,
        )
        
        # Alignment-aware selection
        valid_mask = frame_to_char_idx >= 0
        # ... (既存のselection logic)
        
        disfluency_loss = cross_entropy(selected_logits, selected_labels)
    
    return total_loss, loss_dict
```

## CTCの完全削除

### モデル構造の変更

**旧実装（CTC使用）:**
```python
class MultitaskRNNTModel:
    encoder: Encoder
    decoder: Decoder
    joint_network: JointNetwork
    ctc: CTC  ← これを削除！
    
    def forward():
        encoder_out = encoder(speech)
        ctc_logits = ctc(encoder_out)  ← 削除
        ctc_loss = ctc_loss(ctc_logits, ...)  ← 削除
        
        transducer_loss = ...
        total_loss = transducer_loss + ctc_weight * ctc_loss  ← 削除
```

**新実装（CTC完全削除）:**
```python
class MultitaskRNNTModel:
    encoder: Encoder
    decoder: Decoder
    joint_network: MultitaskJointNetwork
    # ctc: 削除！
    
    def forward():
        encoder_out = encoder(speech)
        decoder_out = decoder(text)
        
        joint_outputs = joint_network(encoder_out, decoder_out)
        asr_logits = joint_outputs["asr_scores"]
        disfluency_logits = joint_outputs["disfluency_scores"]
        
        # RNN-T Loss のみ
        transducer_loss, disfluency_loss = criterion(
            asr_logits, disfluency_logits, ...
        )
        
        total_loss = transducer_loss + λ * disfluency_loss
        # CTC Loss は完全に不要
```

### 設定ファイルの変更

**削除するパラメータ:**
```yaml
# myconf/train_multitask_transducer.yaml

# 削除：
# ctc_weight: 0.3
# use_ctc: true

# Transducerのみ使用
model_conf:
    ctc_weight: 0.0  # または完全に削除
```

## まとめ

### 主要な変更

| 項目 | 旧実装 (CTC依存) | 新実装 (Transducer純粋) |
|------|------------------|------------------------|
| アライメント源 | CTC logits | Transducer joint logits |
| アルゴリズム | 1D Viterbi | 2D Viterbi |
| 関数名 | `viterbi_alignment()` | `transducer_viterbi_alignment()` |
| 入力形状 | (B, T, vocab) | (B, T, U, vocab) |
| 理論的整合性 | ❌ CTCとTransducerで異なるアライメント | ✅ Transducer内で一貫 |
| CTC依存 | ❌ 必須 | ✅ 不要 |

### データフロー比較

**旧実装（CTC使用）:**
```
Encoder → CTC → CTC Loss (補助)
   ↓         ↓
   ↓    Viterbi Alignment → Disfluency (問題！)
   ↓
Decoder → Joint Network → Transducer Loss
```
**問題点:**
- CTCとTransducerで異なるアライメント学習
- 理論的に矛盾

**新実装（CTC完全削除）:**
```
Encoder → Decoder → Joint Network → Transducer Logits
                                         ↓
                                    ┌────┴────┐
                                    ↓         ↓
                              RNN-T Loss   Viterbi Alignment
                                              ↓
                                         Disfluency Loss
```
**利点:**
- 完全にTransducer純粋
- アライメント学習が一貫
- シンプルで理論的に正しい

### 理論的利点

1. **完全な一貫性**: Transducerの学習とdisfluency検出が同じアライメントを使用
2. **純粋なTransducer**: CTC依存を完全に削除、Transducerモデルとして純粋
3. **正確性**: RNN-Tのforward-backward学習で得られる最適アライメントを使用
4. **シンプル**: アライメント取得元が一つ（Transducerのみ）
5. **矛盾の解消**: CTCとTransducerの異なるアライメント学習による矛盾を完全に排除

### コード変更箇所まとめ

**削除が必要なコード:**

1. **multitask_rnnt_model.py**
   - `self.ctc = CTC(...)` の削除
   - `ctc_logits = self.ctc(encoder_out)` の削除
   - `ctc_loss` の計算削除
   - Total lossから`ctc_loss`の項を削除

2. **設定ファイル (train_multitask_transducer.yaml)**
   - `ctc_weight: 0.3` → 削除 or `0.0`
   - `use_ctc: true` → 削除 or `false`

**追加済みのコード（既に実装済み）:**

1. **multitask_joint_network.py**
   - ✅ `transducer_viterbi_alignment()` - Transducer用Viterbiアルゴリズム
   - ✅ `align_disfluency_labels_with_transducer()` - Transducerベースのラベル対応付け
   - ✅ `ctc_logits`パラメータの削除（loss関数から）

この実装により、**完全にCTCフリーのTransducerベースマルチタスク学習**が実現します！
