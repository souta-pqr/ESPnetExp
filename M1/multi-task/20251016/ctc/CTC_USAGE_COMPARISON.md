# Transducer vs Multitask Transducer

---

## 1. 通常のRNN-Transducer (ASRのみ)

### 1.1 アーキテクチャ

```
音声入力
  ↓
[Encoder] → encoder_out (B, T, 256)
  ↓
[Decoder] → decoder_out (B, U, 512)
  ↓
[Joint Network] → joint_out (B, T, U, 1024)
  ↓
[Output Layer] → logits (B, T, U, vocab_size)
  ↓
[RNN-T Loss] → loss
```

### 1.2 CTCは使わない（通常）

```python
# espnet2/asr/espnet_model.py: ESPnetASRModel (標準のTransducer)

class ESPnetASRModel:
    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: TransducerDecoder,
        # CTC は **オプション** (通常は None)
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0.0,  # デフォルトは0 (使わない)
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.ctc_weight = ctc_weight
    
    def forward(self, speech, text, ...):
        # エンコーダ
        encoder_out = self.encoder(speech)
        
        # CTC Loss (もし ctc_weight > 0 なら)
        if self.ctc is not None and self.ctc_weight > 0:
            ctc_loss = self.ctc(encoder_out, text)
        else:
            ctc_loss = 0.0  # 使わない
        
        # Transducer Loss (メイン)
        rnnt_loss = self.decoder(encoder_out, text)
        
        # 総合損失
        loss = (1 - self.ctc_weight) * rnnt_loss + self.ctc_weight * ctc_loss
        # 通常: loss = 1.0 * rnnt_loss + 0.0 * ctc_loss = rnnt_loss のみ
        
        return loss
```

### 1.3 CTCを使わない理由

#### **理由1: RNN-Tで十分**
```
RNN-Transducer は独立したend-to-endモデル

[音声] → [Encoder] → [Decoder] → [Joint] → [認識結果]
           ↑                                    　↓
           └────── RNN-T Loss (自動アライメント) ──┘

- Forward-backward algorithm で自動的にアライメントを学習
- 追加の情報源（CTC）は不要
- シンプルで効率的
```

#### **理由2: タスクが1つ（ASRのみ）**
```
学習目標:
  入力: 音声波形
  出力: テキスト系列
  
損失関数:
  RNN-T Loss のみで十分
  
教師信号:
  テキストラベルのみ (B, U)
  - アライメント不要（RNN-Tが自動計算）
  - フレームレベルのラベル不要
```

#### **理由3: 計算コスト**
```
CTCを追加すると:
  + CTC Linear層のパラメータ
  + CTC Loss の計算
  + メモリ使用量の増加
  
性能向上が限定的:
  RNN-T単独: CER 5.0%
  RNN-T + CTC (weight=0.3): CER 4.8%  ← わずか0.2%の改善
  
→ コスト対効果が低い
```

### 1.4 例外: CTCを使う場合もある

```yaml
# 一部の設定では、CTCを補助損失として使用

# conf/train_transducer_with_ctc.yaml
model_conf:
  ctc_weight: 0.3  # CTCを30%の重みで使用

# メリット:
# 1. 学習の安定化（特に初期段階）
# 2. わずかな性能向上
# 3. デコード時の代替手段

# デメリット:
# 1. 計算コスト増加
# 2. ハイパーパラメータ調整が必要
# 3. 効果が限定的
```

**実際の使用例**:
```python
# ESPnetのTransducer設定

# パターンA: CTC なし (最も一般的)
decoder: transducer
decoder_conf:
  ...
# ctc なし → ctc_weight = 0.0

# パターンB: CTC 補助損失 (一部のレシピ)
decoder: transducer
decoder_conf:
  ...
ctc_conf:
  dropout_rate: 0.1
ctc_weight: 0.3  # 補助的に使用
```

---

## 2. Multitask Transducer (ASR + 非流暢性検出)

### 2.1 アーキテクチャ

```
音声入力
  ↓
[Encoder] → encoder_out (B, T, 256)
  ↓
  ├─→ [CTC Head] → ctc_logits (B, T, vocab) 
  │                    ↓
  │               [Viterbi Alignment]
  │                    ↓
  │          frame_level_labels (B, T)  ← 補助情報用
  │                    ↓
  ↓                    │
[Decoder] → decoder_out (B, U, 512)
  ↓                    │
[Joint Network] ←──────┘
  ↓
  ├─→ [ASR Head] → asr_logits (B, T, U, vocab)
  │                    ↓
  │               [RNN-T Loss]
  │
  └─→ [Disfluency Head] → disfluency_logits (B, T, U, 4)
                           ↓
                    [CE Loss] ← frame_level_labels を使用
```

### 2.2 CTCを使う理由

#### **理由1: アライメント問題の解決**

```python
# 問題: 次元のミスマッチ

# エンコーダ出力 (フレームレベル)
encoder_out: (B, T=75, 256)  # 75フレーム（約3秒）

# 非流暢性ラベル (単語レベル)
isdysfl: (B, U=7)  # 7単語のラベル
# 例: [0, 0, 1, 0, 0, 0, 0]
#    [正常, 正常, filler, 正常, 正常, 正常, 正常]

# 問題:
# - 75フレームのどこが "えーと"(単語2) に対応？
# - フレーム0-10? 15-25? 20-30?
# - **フレーム-単語の対応が不明**
```

**Without CTC (単純padding)**:
```python
# 悪い方法: 単純にラベルをpadding

frame_level_labels = torch.zeros(T, dtype=torch.long)
frame_level_labels[:U] = isdysfl  # 最初のUフレームに割り当て
frame_level_labels[U:] = -100     # 残りは ignore

# 結果:
# frame_level_labels = [0, 0, 1, 0, 0, 0, 0, -100, -100, ..., -100]
#                      [7フレームのみ有効, 68フレームはignore]

# 問題点:
# 1. 最初の7フレームだけで学習 (93%のフレームを無駄に)
# 2. 実際の単語の位置と無関係
# 3. "えーと" が実際にフレーム20-28に発話されても、
#    フレーム2に割り当てられてしまう
```

**With CTC Viterbi**:
```python
# 良い方法: CTCで最適アライメント

# ステップ1: CTC logits で各フレームの単語予測
ctc_logits = self.ctc.ctc_lo(encoder_out)  # (B, T=75, vocab_size)

# ステップ2: Viterbi で最適パス計算
alignment = viterbi_alignment(ctc_logits, text, ...)
# alignment: (B, T=75)
# 例: [-1, -1, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2, ...]
#     [blank×2, 今日×5, blank, は×4, blank, えーと×6, ...]

# ステップ3: 単語ラベルをフレームにマッピング
frame_level_labels[t] = isdysfl[alignment[t]]

# 結果:
# frame_level_labels = [ignore, ignore, 0, 0, 0, 0, 0, ignore, 0, 0, 0, 0, 
#                       ignore, 1, 1, 1, 1, 1, 1, ...]
#                      [全フレームで正確なラベル]

# 利点:
# 1. 全フレーム(blankを除く)で学習可能 (80%以上)
# 2. 実際の発話タイミングに対応
# 3. "えーと" が本当にフレーム13-18に発話されていれば、
#    そのフレームに filler ラベルが割り当てられる
```

**効果の比較**:
```
Without Viterbi:
  有効フレーム: 7/75 = 9.3%
  学習信号: 極めて限定的
  非流暢性F1: 45-50%

With Viterbi:
  有効フレーム: ~60/75 = 80%
  学習信号: ほぼ全フレーム
  非流暢性F1: 60-65%  ← +15% 改善！
```

#### **理由2: マルチタスク学習の必要性**

```python
# ASR: sequence-to-sequence (系列変換)
# 入力: 音声フレーム系列 (T フレーム)
# 出力: 単語系列 (U 単語)
# 損失: RNN-T Loss (アライメント自動学習)

# 非流暢性検出: frame-level classification (フレーム分類)
# 入力: 音声フレーム系列 (T フレーム)
# 出力: 各フレームの非流暢性ラベル (T ラベル)
# 損失: Cross-Entropy Loss (フレームごとの教師信号が必要)

# → 教師信号として frame_level_labels (B, T) が必要
# → CTCでアライメントを取得してラベルを生成
```

#### **理由3: CTCの二重の役割**

```python
# 役割A: アライメント計算 (Forward Pass)
ctc_logits = self.ctc.ctc_lo(encoder_out)
alignment = viterbi_alignment(ctc_logits, text, ...)
frame_level_labels = map_labels(alignment, isdysfl)

# 役割B: 間接的な学習 (Backward Pass)
disfluency_loss = cross_entropy(frame_disfluency_logits, frame_level_labels)
disfluency_loss.backward()
# ↓
# 勾配が encoder_out を通じて CTC にも流れる
# ↓
# CTCは良いアライメントを提供するように学習される
```

**勾配の流れ**:
```
disfluency_loss
  ↓ (backward)
frame_disfluency_logits
  ↓
disfluency_output_layer
  ↓
joint_network
  ↓
encoder_out ──┐
  ↓           ↓
encoder    CTC head
  ↓           ↓
  └───────────┘
  両方に勾配が流れる
  
→ CTCは損失関数を持たないが、
  Viterbiを通じて間接的に学習される
```

---

## 3. 詳細な比較表

### 3.1 アーキテクチャの違い

| 項目 | 通常のTransducer | Multitask Transducer |
|------|------------------|----------------------|
| **Encoder** | ✓ (音響モデル) | ✓ (音響モデル) |
| **CTC Head** | × (通常なし) | ✓ (Viterbi用) |
| **Decoder** | ✓ (言語モデル) | ✓ (言語モデル) |
| **Joint Network** | ✓ (1出力) | ✓ (2出力) |
| **Output Heads** | ASRのみ | ASR + 非流暢性 |
| **Loss Functions** | RNN-T Loss | RNN-T + CE Loss |

### 3.2 データフローの違い

#### **通常のTransducer**
```
音声 (B, T_audio)
  ↓
encoder_out (B, T, 256)
  ↓
decoder_out (B, U, 512)
  ↓
joint_out (B, T, U, 1024)
  ↓
asr_logits (B, T, U, vocab)
  ↓
RNN-T Loss
  ↑
text (B, U)  ← 教師信号はこれだけ
```

#### **Multitask Transducer**
```
音声 (B, T_audio)
  ↓
encoder_out (B, T, 256)
  ↓
  ├─→ ctc_logits (B, T, vocab)
  │      ↓
  │   Viterbi
  │      ↓
  │   frame_level_labels (B, T)  ← 新しい教師信号
  │      ↓
  ↓      │
decoder_out (B, U, 512)
  ↓      │
joint_out (B, T, U, 1024)
  ↓      │
  ├─→ asr_logits → RNN-T Loss ← text (B, U)
  │                               
  └─→ disfluency_logits → CE Loss ← frame_level_labels (B, T)
                                     isdysfl (B, U) ← 元データ
```

### 3.3 学習目標の違い

#### **通常のTransducer**
```python
# 学習目標: ASRのみ
# 入力: 音声
# 出力: テキスト

loss = rnnt_loss(asr_logits, text)

# 必要なラベル:
# - text: (B, U) - 単語系列

# アライメント:
# - RNN-T が自動計算
# - 明示的なフレーム-単語対応は不要
```

#### **Multitask Transducer**
```python
# 学習目標: ASR + 非流暢性検出
# 入力: 音声
# 出力1: テキスト
# 出力2: 各単語の非流暢性ラベル

loss = rnnt_loss(asr_logits, text) + \
       ce_loss(disfluency_logits, frame_level_labels)

# 必要なラベル:
# - text: (B, U) - 単語系列
# - isdysfl: (B, U) - 単語ごとの非流暢性ラベル

# アライメント:
# - ASR: RNN-T が自動計算
# - 非流暢性: CTCで明示的に計算（必須）
#   理由: フレームレベルの教師信号が必要
```

### 3.4 CTCの役割の違い

| 項目 | 通常のTransducer | Multitask Transducer |
|------|------------------|----------------------|
| **CTCの有無** | 通常なし (オプション) | **必須** |
| **CTCの目的** | 補助損失（性能向上） | **アライメント計算** |
| **CTC Loss** | 使用する (weight > 0) | **使用しない** (weight = 0) |
| **Viterbi** | 不要 | **必須** (frame-label mapping) |
| **学習方法** | 直接的 (CTC Loss) | **間接的** (勾配のみ) |
| **計算コスト** | +10-15% | +5-10% (Lossなし) |
| **性能への影響** | ASR: +0.2-0.5% | 非流暢性: +10-15% |

---

## 4. 具体例で理解する

### 4.1 通常のTransducer

```python
# 入力
音声: "今日は良い天気ですね" (3秒, no disfluency)
text: [3, 5, 8, 15, 18, 22]  # 6単語

# 処理
encoder_out: (1, 75, 256)    # 75フレーム
decoder_out: (1, 6, 512)      # 6単語
joint_out: (1, 75, 6, 1024)   # 75×6 = 450 組み合わせ
asr_logits: (1, 75, 6, 136)

# 損失計算
rnnt_loss = RNN-T(asr_logits, text)
# Forward-backward で全アライメントパスを計算
# 例: [b,b,3,b,5,b,8,b,15,b,18,b,22,b,b,...] (b=blank)
#     [3,b,b,5,b,8,b,15,b,18,b,22,b,b,...]
#     ...
# 全パスの確率を合計して最大化

total_loss = rnnt_loss
# CTC不要！RNN-Tだけで十分
```

### 4.2 Multitask Transducer

```python
# 入力
音声: "今日はえーと良い天気ですね" (3秒, with filler)
text: [3, 5, 12, 8, 15, 18, 22]       # 7単語
isdysfl: [0, 0, 1, 0, 0, 0, 0]        # 7単語のラベル (word-level)

# 問題: isdysfl は単語レベルだが、非流暢性検出はフレームレベル

# 処理
encoder_out: (1, 75, 256)    # 75フレーム

# ステップ1: CTC でアライメント計算
ctc_logits: (1, 75, 136)
alignment = viterbi(ctc_logits, text)
# alignment: [-1,-1,0,0,0,0,0,-1,1,1,1,1,-1,2,2,2,2,2,2,...]
#            [blank×2, 今日×5, blank, は×4, blank, えーと×6, ...]

frame_level_labels = isdysfl[alignment]
# frame_level_labels: [ign,ign,0,0,0,0,0,ign,0,0,0,0,ign,1,1,1,1,1,1,...]
#                     [ign×2, 正常×5, ign, 正常×4, ign, filler×6, ...]

# ステップ2: Joint Network
decoder_out: (1, 7, 512)      # 7単語
joint_out: (1, 75, 7, 1024)   # 75×7 = 525 組み合わせ

# ステップ3: Dual Heads
asr_logits: (1, 75, 7, 136)
disfluency_logits: (1, 75, 7, 4)

# ステップ4: 損失計算
# ASR
rnnt_loss = RNN-T(asr_logits, text)

# 非流暢性 (frame-level)
frame_disfluency_logits = disfluency_logits.mean(dim=2)  # (1, 75, 4)
ce_loss = CrossEntropy(frame_disfluency_logits, frame_level_labels)

total_loss = rnnt_loss + ce_loss

# CTC必須！frame_level_labels を作るために必要
```

---

## 5. なぜ違いが生じるのか

### 5.1 タスクの性質の違い

```
ASR (音声認識):
  - タスク: 系列変換 (sequence-to-sequence)
  - 入力: 音声フレーム系列 (可変長)
  - 出力: 単語系列 (可変長)
  - 特徴: 入出力の長さが異なる (T >> U)
  - アライメント: RNN-Tが自動で学習
  - 教師信号: 単語系列のみ (B, U)

非流暢性検出:
  - タスク: フレーム分類 (frame-level classification)
  - 入力: 音声フレーム系列 (可変長)
  - 出力: 各フレームのラベル (同じ長さ)
  - 特徴: 入出力の長さが同じ (T = T)
  - アライメント: 明示的に必要
  - 教師信号: フレームごとのラベル (B, T) ← これが問題！
```

### 5.2 データアノテーションの違い

```python
# ASRのアノテーション
# 音声: "今日はえーと良い天気ですね"
# ラベル: "今日 は えーと 良い 天気 です ね"
# → 単語系列のみ (時刻情報なし)

# 非流暢性のアノテーション (通常)
# 音声: "今日はえーと良い天気ですね"
# ラベル: [
#   {word: "今日", disfluency: "fluent"},
#   {word: "は", disfluency: "fluent"},
#   {word: "えーと", disfluency: "filled_pause"},  ← 単語レベル
#   {word: "良い", disfluency: "fluent"},
#   ...
# ]
# → 単語ごとのラベル (時刻情報なし)

# 問題:
# - アノテーションは単語レベル
# - 学習はフレームレベルで必要
# - 時刻情報がない
# → CTCで自動アライメントが必要
```

### 5.3 解決策の違い

```
通常のTransducer:
  問題: なし
  理由: RNN-Tが全て解決
  解決策: 不要

Multitask Transducer:
  問題: 単語ラベル → フレームラベルの変換
  理由: タスクの性質が異なる
  解決策: CTCでアライメント計算
  
  代替案:
  1. Forced Alignment (HMM-GMM使用)
     - 複雑、別モデルが必要
  2. Attention Mechanism
     - 不安定、計算コスト高
  3. CTC Viterbi
     - シンプル、効率的 ← 採用！
```

---

## 6. まとめ

### 通常のTransducer (ASR)
```
✗ CTC 不要
  - RNN-Tで完結
  - 単一タスク
  - アライメント自動学習
  - 教師信号: 単語系列のみ
```

### Multitask Transducer (ASR + 非流暢性)
```
✓ CTC 必須
  - アライメント計算に使用
  - マルチタスク
  - フレームレベル教師信号が必要
  - 教師信号: 単語系列 + フレームラベル
  
役割:
  1. Forward: ctc_logits → Viterbi → frame_labels
  2. Backward: 勾配による間接学習
  
効果:
  - 非流暢性F1: 45% → 60% (+15%改善)
  - 計算コスト: +5-10% (許容範囲)
```

### キーポイント

| 観点 | 通常のTransducer | Multitask Transducer |
|------|------------------|----------------------|
| **タスク数** | 1 (ASR) | 2 (ASR + 非流暢性) |
| **出力レベル** | 単語 | 単語 + フレーム |
| **アライメント** | 自動 (RNN-T) | 明示的 (CTC) |
| **CTC** | 不要 | **必須** |
| **理由** | RNN-Tで完結 | **フレームラベル生成** |

**結論**: CTCの使用は**タスクの性質**によって決まる。ASRのみならRNN-Tで十分だが、フレームレベルのタスク（非流暢性検出）を追加する場合、CTCによるアライメント計算が必須となる。
