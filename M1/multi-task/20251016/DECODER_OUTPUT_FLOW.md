# デコーダ周りの出力フロー詳細

## 概要

1. **Prediction Network (RNN Decoder)**: 単語系列から言語表現を生成
2. **Joint Network**: エンコーダとデコーダの出力を結合
3. **Dual Output Heads**: ASR と 補助情報検出の2つのタスク

---

## 入力データ (具体例)

```python
# バッチサイズ B=2 の例
batch = {
    # エンコーダ出力 (音響表現)
    'encoder_out': torch.Tensor,        # (2, T, 256)
    'encoder_out_lens': torch.Tensor,   # [75, 68] - 各発話のフレーム数
    
    # ASRラベル (Ground Truth)
    'text': torch.Tensor,               # (2, L_max)
    'text_lengths': torch.Tensor,       # [7, 5] - 各発話の単語数
    
    # 補助情報ラベル (Ground Truth)
    'isdysfl': torch.Tensor,            # (2, L_max)
    'isdysfl_lengths': torch.Tensor,    # [7, 5]
    
    # CTC logits (Viterbi用)
    'ctc_logits': torch.Tensor,         # (2, T, vocab_size)
}

# 具体的な値
# サンプル0: "今日はえーと良い天気ですね" (15文字, 75フレーム)
# カタカナ表記: "キョウワエートイイテンキデスネ"
text[0] = [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78, 0, ...]  # padding
isdysfl[0] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]    
# [キ, ョ, ウ, ワ, エ, ー, ト, イ, イ, テ, ン, キ, デ, ス, ネ]

# サンプル1: "そのー今日は晴れ" (10文字, 68フレーム)
# カタカナ表記: "ソノーキョウワハレ"
text[1] = [28, 71, 9, 23, 45, 12, 67, 84, 92, 0, ...]
isdysfl[1] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...]
# [ソ, ノ, ー, キ, ョ, ウ, ワ, ハ, レ]
```

---

## Phase 1: Prediction Network (RNN Decoder)

### ファイル: `espnet2/asr/decoder/multitask_transducer.py`

### 1.1 入力処理

```python
def forward(
    self,
    encoder_out: torch.Tensor,          # (B=2, T_max=75, 256)
    encoder_out_lens: torch.Tensor,     # [75, 68]
    labels: torch.Tensor,               # (B=2, L_max=7)
    labels_lengths: torch.Tensor,       # [7, 5]
    isdysfl: Optional[torch.Tensor],    # (B=2, L_max=7)
    isdysfl_lengths: Optional[torch.Tensor],  # [7, 5]
    ctc_logits: Optional[torch.Tensor], # (B=2, T_max=75, vocab_size=136)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Multitask Transducer Decoder の forward pass
    """
    
    # バッチ情報
    batch_size = encoder_out.size(0)  # 2
    T_max = encoder_out.size(1)       # 75 (最大フレーム数)
    L_max = labels.size(1)            # 7 (最大単語数)
```

### 1.2 Embedding Layer

```python
    # ステップ1: 単語ID → Embedding
    # labels: (B=2, L_max=7)
    # 例: labels[0] = [23, 45, 12, 67, 8, 9, 15]  (キ, ョ, ウ, ワ, エ, ー, ト)
    #     labels[1] = [33, 33, 55, 67, 23, 0, 0]  (イ, イ, テ, ン, キ, padding...)
    
    decoder_in = self.embed(labels)
    # decoder_in: (B=2, L_max=7, embed_size=256)
    
    # 各文字IDが256次元ベクトルに変換される
    # decoder_in[0, 0, :] = embed(23) → [0.12, -0.45, 0.78, ...] (256次元) "キ"
    # decoder_in[0, 1, :] = embed(45) → [-0.23, 0.67, -0.12, ...]         "ョ"
    # decoder_in[0, 2, :] = embed(12) → [0.89, -0.34, 0.56, ...]          "ウ"
    # ...
```

**可視化: Embedding の意味**
```
文字ID → Embedding空間

23 (キ)   → [0.12, -0.45, 0.78, ..., 0.34]  (256次元)
45 (ョ)   → [-0.23, 0.67, -0.12, ..., 0.89]
12 (ウ)   → [0.89, -0.34, 0.56, ..., -0.12]
8  (エ)   → [0.45, -0.67, 0.23, ..., 0.56]
9  (ー)   → [0.51, -0.62, 0.18, ..., 0.49]
```

### 1.3 Dropout on Embeddings

```python
    # Dropout (学習時のみ、dropout_embed=0.1)
    if self.training:
        decoder_in = F.dropout(decoder_in, p=self.dropout_embed)
    # decoder_in: (B=2, L_max=7, 256)
    
    # 一部のembedding値をランダムに0にして過学習を防ぐ
```

### 1.4 LSTM処理

```python
    # ステップ2: LSTM (2層、hidden_size=512)
    # RNN Transducerでは、各単語位置での予測を独立に計算
    
    # 初期hidden state
    h0 = torch.zeros(
        self.num_layers,     # 2
        batch_size,          # 2
        self.hidden_size,    # 512
        device=decoder_in.device
    )
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=decoder_in.device)
    
    # LSTM forward
    decoder_out, (hn, cn) = self.rnn(decoder_in, (h0, c0))
    # decoder_out: (B=2, L_max=7, hidden_size=512)
    # hn: (num_layers=2, B=2, 512) - 最終hidden state
    # cn: (num_layers=2, B=2, 512) - 最終cell state
```

**LSTM の詳細処理**
```python
# 時系列方向の処理

for t in range(L_max):  # 各単語位置
    # 入力
    x_t = decoder_in[:, t, :]  # (B=2, 256)
    
    # LSTM cell (Layer 1)
    i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_prev[0] + b_hi)  # input gate
    f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_prev[0] + b_hf)  # forget gate
    g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_prev[0] + b_hg)     # cell gate
    o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_prev[0] + b_ho)  # output gate
    
    c_t[0] = f_t * c_prev[0] + i_t * g_t  # new cell state
    h_t[0] = o_t * tanh(c_t[0])           # new hidden state
    
    # LSTM cell (Layer 2)
    # h_t[0] を入力として同様の処理
    # ...
    
    # 出力
    decoder_out[:, t, :] = h_t[1]  # Layer 2 の hidden state
```

**Decoder Output の意味**
```python
# decoder_out: (B=2, L_max=7, 512)

# サンプル0
decoder_out[0, 0, :] = h_0  # 文字0 "キ" までの文脈表現 (512次元)
decoder_out[0, 1, :] = h_1  # 文字1 "ョ" までの文脈表現 (キョ)
decoder_out[0, 2, :] = h_2  # 文字2 "ウ" までの文脈表現 (キョウ)
decoder_out[0, 3, :] = h_3  # 文字3 "ワ" までの文脈表現 (キョウワ)
decoder_out[0, 4, :] = h_4  # 文字4 "エ" までの文脈表現 (キョウワエ)
decoder_out[0, 5, :] = h_5  # 文字5 "ー" までの文脈表現 (キョウワエー)
decoder_out[0, 6, :] = h_6  # 文字6 "ト" までの文脈表現 (キョウワエート)
...

# 各ベクトルは「これまでの文字列」の情報を保持
# h_2 は "キョウ" (今日) の文脈を持つ
# h_6 は "キョウワエート" (今日はえーと) の文脈を持つ
```

### 1.5 Dropout on LSTM Output

```python
    # Dropout (学習時のみ、dropout=0.1)
    if self.training and self.dropout > 0:
        decoder_out = F.dropout(decoder_out, p=self.dropout)
    # decoder_out: (B=2, L_max=7, 512)
```

---

## Phase 2: Joint Network

### ファイル: `espnet2/asr/layers/multitask_joint_network.py`

### 2.1 入力の準備

```python
def forward(
    self,
    encoder_out: torch.Tensor,  # (B=2, T_max=75, encoder_dim=256)
    decoder_out: torch.Tensor,  # (B=2, L_max=7, decoder_dim=512)
) -> Dict[str, torch.Tensor]:
    """
    Joint Network: エンコーダとデコーダの出力を結合
    """
    
    batch_size = encoder_out.size(0)  # 2
    T = encoder_out.size(1)           # 75
    U = decoder_out.size(1)           # 7
```

### 2.2 次元拡張とブロードキャスト

```python
    # ステップ1: 次元を拡張して全組み合わせを作成
    
    # エンコーダ出力: (B, T, D_enc) → (B, T, 1, D_enc)
    encoder_out_expanded = encoder_out.unsqueeze(2)
    # (2, 75, 1, 256)
    
    # デコーダ出力: (B, U, D_dec) → (B, 1, U, D_dec)
    decoder_out_expanded = decoder_out.unsqueeze(1)
    # (2, 1, 7, 512)
```

**ブロードキャストの可視化**
```
encoder_out_expanded: (2, 75, 1, 256)
    ├─ サンプル0, フレーム0,  単語位置*,  音響表現
    ├─ サンプル0, フレーム1,  単語位置*,  音響表現
    ├─ ...
    └─ サンプル0, フレーム74, 単語位置*,  音響表現

decoder_out_expanded: (2, 1, 7, 512)
    ├─ サンプル0, フレーム*, 単語位置0, 言語表現
    ├─ サンプル0, フレーム*, 単語位置1, 言語表現
    ├─ ...
    └─ サンプル0, フレーム*, 単語位置6, 言語表現

* は全てに適用されることを示す (ブロードキャスト)

結合後: (2, 75, 7, joint_space_size)
    全てのフレーム×単語の組み合わせ
    = 2 × 75 × 7 = 1050 個の組み合わせ
```

### 2.3 Joint空間への投影

```python
    # ステップ2: 線形投影
    
    # Encoder projection: 256 → 1024
    encoder_proj = self.encoder_proj(encoder_out_expanded)
    # (B=2, T=75, 1, joint_space_size=1024)
    
    # Decoder projection: 512 → 1024
    decoder_proj = self.decoder_proj(decoder_out_expanded)
    # (B=2, 1, U=7, joint_space_size=1024)
```

### 2.4 加算と活性化

```python
    # ステップ3: 加算 (ブロードキャストで全組み合わせ)
    joint_out = encoder_proj + decoder_proj
    # (B=2, T=75, U=7, joint_space_size=1024)
    
    # 活性化関数
    joint_out = torch.tanh(joint_out)
    # (B=2, T=75, U=7, 1024)
```

**Joint Output の詳細構造**
```python
# joint_out[b, t, u, :] の意味:
# バッチ b, フレーム t で単語位置 u を出力する場合の統合表現

# 例: joint_out[0, 17, 2, :]
# = サンプル0, フレーム17 で "えーと"(位置2) を出力する表現 (1024次元)
# = encoder_out[0, 17, :] と decoder_out[0, 2, :] の統合

# 具体例:
joint_out[0, 0, 0, :] = tanh(enc[0,0] + dec[0,0])  # フレーム0で単語0
joint_out[0, 0, 1, :] = tanh(enc[0,0] + dec[0,1])     # フレーム0で単語1
...
joint_out[0, 17, 2, :] = tanh(enc[0,17] + dec[0,2])  # フレーム17で単語2
...
joint_out[0, 74, 6, :] = tanh(enc[0,74] + dec[0,6])  # フレーム74で単語6
```

### 2.5 Dropout

```python
    # Dropout (学習時のみ)
    if self.training and self.dropout_rate > 0:
        joint_out = F.dropout(joint_out, p=self.dropout_rate)
    # joint_out: (B=2, T=75, U=7, 1024)
```

---

## Phase 3: Dual Output Heads

### 3.1 ASR Output Head

```python
    # ステップ4-A: ASR予測ヘッド
    # joint_out: (B=2, T=75, U=7, 1024)
    
    asr_logits = self.asr_output_layer(joint_out)
    # Linear: 1024 → vocab_size (136)
    # asr_logits: (B=2, T=75, U=7, vocab_size=136)
```

**ASR Logits の意味**
```python
# asr_logits[b, t, u, v] = バッチb, フレームt, 文字位置u で文字v を出力するスコア

# 例: asr_logits[0, 17, 2, :]
# = フレーム17, 文字位置2 での全文字の予測スコア (136次元)
# = [0.5, 1.2, 0.8, 8.5, ..., 0.3]  (文字ID 0〜135 のスコア)
#              ↑
#         文字ID=12 "ウ" が高スコア

# 全体の形状: (2, 75, 15, 136)
# 意味: 各 (フレーム, 文字位置) の組み合わせで、どの文字を出力するかのスコア
```

**ASR予測の具体例**
```python
# サンプル0, フレーム17, 文字位置2 "ウ" での予測
logits = asr_logits[0, 17, 2, :]  # (136,)
probs = F.softmax(logits, dim=-1)

# Top-5 予測
# 文字ID | スコア | 確率   | 文字
# -------|--------|--------|--------
# 12     | 8.5    | 0.921  | ウ   ← 正解！
# 23     | 3.2    | 0.045  | キ
# 45     | 2.8    | 0.030  | ョ
# 67     | 1.5    | 0.008  | ワ
# 0      | -2.1   | 0.002  | blank
```

### 3.2 Disfluency Output Head

```python
    # ステップ4-B: 補助情報予測ヘッド
    if self.use_disfluency_detection:
        
        # オプション: Projection Layer (次元削減)
        if self.disfluency_projection is not None:
            disfluency_features = self.disfluency_projection(joint_out)
            # Linear: 1024 → projection_size (128)
            # disfluency_features: (B=2, T=75, U=15, 128)
        else:
            disfluency_features = joint_out
            # (B=2, T=75, U=15, 1024)
        
        # 補助情報分類層
        disfluency_logits = self.disfluency_output_layer(disfluency_features)
        # Linear: 128 → disfluency_classes (4)
        # disfluency_logits: (B=2, T=75, U=15, disfluency_classes=4)
```

**Disfluency Logits の意味**
```python
# disfluency_logits[b, t, u, c] = バッチb, フレームt, 文字位置u で補助情報クラスc のスコア

# 4つのクラス:
# 0 = fluent (流暢)
# 1 = filler (フィラー: えー、あのー、そのー等の文字)
# 2 = repetition (繰り返し: 今日今日、そそうだね等の文字)
# 3 = interjection (感動詞: うん、はい、あ等の文字)

# 例: disfluency_logits[0, 17, 4, :]
# = フレーム17, 文字位置4 "エ" での補助情報予測 (4次元)
# = [1.2, 8.5, 0.8, 1.1]
#    ↑    ↑    ↑    ↑
#  fluent filler rep other
```

**補助情報予測の具体例**
```python
# サンプル0, フレーム17, 文字位置4 "エ" での予測
# "キ ョ ウ ワ エ ー ト" の "エ" (位置4はフィラーの開始)
logits = disfluency_logits[0, 17, 4, :]  # (4,)
probs = F.softmax(logits, dim=-1)

# クラス | スコア | 確率   | 意味
# -------|--------|--------|------------
# 0      | 1.2    | 0.018  | fluent
# 1      | 8.5    | 0.963  | filler
# 2      | 0.8    | 0.012  | repetition
# 3      | 1.1    | 0.016  | interjection

# Ground Truth: isdysfl[0, 4] = 1 (filled_pause) - "エ"はフィラー
# 予測: argmax = 1
```

### 3.3 出力の整理

```python
    # 戻り値
    return {
        "asr_logits": asr_logits,              # (B=2, T=75, U=7, 136)
        "disfluency_logits": disfluency_logits, # (B=2, T=75, U=7, 4)
    }
```

---

## Phase 4: 損失計算への入力

### ファイル: `espnet2/asr/layers/multitask_joint_network.py: MultitaskTransducerLoss`

```python
def forward(
    self,
    encoder_out: torch.Tensor,      # (B=2, T=75, 256)
    encoder_out_lens: torch.Tensor, # [75, 68]
    decoder_out: torch.Tensor,      # (B=2, U=15, 512) 
    labels: torch.Tensor,           # (B=2, U=15) - ASR labels
    labels_lengths: torch.Tensor,   # [15, 9]  # 文字数
    isdysfl: torch.Tensor,          # (B=2, U=15) - Disfluency labels
    isdysfl_lengths: torch.Tensor,  # [15, 9]
    ctc_logits: torch.Tensor,       # (B=2, T=75, 136)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    損失計算
    """
    
    # ステップ1: Joint Network で logits を取得
    joint_outputs = self.joint_network(encoder_out, decoder_out)
    asr_logits = joint_outputs["asr_logits"]              # (B=2, T=75, U=15, 136)
    disfluency_logits = joint_outputs["disfluency_logits"] # (B=2, T=75, U=15, 4)
```

### 4.1 RNN-T Loss (ASR)

```python
    # ステップ2: RNN-Transducer Loss
    asr_loss = self.rnnt_loss(
        logits=asr_logits,           # (B=2, T=75, U=15, 136)
        targets=labels,              # (B=2, U=15) 
        logit_lengths=encoder_out_lens,  # [75, 68]
        target_lengths=labels_lengths,   # [15, 9]
        blank=self.blank_id,         # 0
        reduction="mean",
    )
    # asr_loss: スカラー (例: 2.45)
```

**RNN-T Loss の詳細**
```python
# Forward-Backward algorithm で全アライメントパスを計算

# サンプル0: "キョウワエートイイテンキデスネ" (T=75, U=15文字)
# 正解ラベル: [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78]
#            [キ, ョ, ウ, ワ, エ, ー, ト, イ, イ, テ, ン, キ, デ, ス, ネ]

# 可能なアライメントパスの例:
# パス1: [b,b,23,b,45,b,b,12,b,67,b,8,b,9,b,15,b,33,b,33,b,...] (b=blank)
# パス2: [b,23,23,b,45,b,12,12,b,67,b,8,b,9,b,15,b,33,b,33,b,...]
# パス3: [23,b,b,45,b,12,b,b,67,b,8,b,9,b,15,b,33,b,33,b,...]
# ...

# 全パスの確率を合計:
# P(Y|X) = Σ P(π|X) for all valid paths π
# Loss = -log P(Y|X)

# asr_logits[b, t, u, v] のスコアを使って各パスの確率を計算
# 各文字が適切なフレームに割り当てられるように学習
```

### 4.2 Viterbi Alignment

```python
    # ステップ3: Viterbi Alignment (補助情報用)
    if self.use_viterbi_alignment and ctc_logits is not None:
        
        # CTC logits から最適アライメントを計算
        frame_level_labels = align_disfluency_labels_with_viterbi(
            encoder_out=encoder_out,          # (B=2, T=75, 256)
            encoder_out_lens=encoder_out_lens, # [75, 68]
            text=labels,                      # (B=2, U=15)  # 文字単位
            text_lengths=labels_lengths,      # [15, 9]  # 文字数
            isdysfl=isdysfl,                  # (B=2, U=15)  # 文字レベルラベル
            ctc_logits=ctc_logits,            # (B=2, T=75, 136)
            blank_id=self.blank_id,
        )
        # frame_level_labels: (B=2, T_max=75)
        
        # サンプル0の結果: "キョウワエートイイテンキデスネ"
        # labels[0] = [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78]
        #            [キ, ョ, ウ, ワ, エ, ー, ト, イ, イ, テ, ン, キ, デ, ス, ネ]
        # isdysfl[0] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # frame_level_labels[0] = 
        # [-100, -100, 0, 0, 0, 0, 0,    # フレーム0-6: "キ" (fluent)
        #  -100, 0, 0, 0, 0,              # フレーム7-11: "ョ" (fluent)
        #  -100, 0, 0, 0, 0,              # フレーム12-16: "ウ" (fluent)
        #  -100, 0, 0, 0,                 # フレーム17-20: "ワ" (fluent)
        #  -100, 1, 1, 1, 1, 1,           # フレーム21-26: "エ" (filler)
        #  -100, 1, 1, 1, 1, 1,           # フレーム27-32: "ー" (filler)
        #  -100, 1, 1, 1, 1, 1, ...]      # フレーム33-38: "ト" (filler)
        #
        # 各文字が複数フレームにマッピングされる
        # blankフレーム(-100)と文字フレーム(0 or 1)が交互に現れる
```

### 4.3 Frame-level Prediction

```python
    # ステップ4: Disfluency logits を Frame-level に変換
    # disfluency_logits: (B=2, T=75, U=15, 4)
    
    # 方法: 文字次元 (U) で平均 pooling
    frame_disfluency_logits = disfluency_logits.mean(dim=2)
    # frame_disfluency_logits: (B=2, T=75, 4)
```

**Frame-level Logits の意味**
```python
# frame_disfluency_logits[b, t, :] = バッチb, フレームt での補助情報予測

# サンプル0, フレーム25 での予測 (フィラー "エ" の領域):
logits = frame_disfluency_logits[0, 25, :]  # (4,)
# = mean(disfluency_logits[0, 25, :, :])  # 全15文字位置で平均
# = [1.2, 8.5, 0.8, 1.1]
#    ↑    ↑    ↑    ↑
#  fluent filler rep interjection

# Ground Truth: frame_level_labels[0, 25] = 1 (filler) - "エ"のフレーム
# 予測: argmax = 1 → 正しく検出！

# サンプル0, フレーム10 での予測
logits = frame_disfluency_logits[0, 10, :]  # (4,)
# = [8.2, 1.1, 0.6, 0.8]
#    ↑    ↑    ↑    ↑
#  fluent filler rep interjection

# Ground Truth: frame_level_labels[0, 10] = 0 (fluent) - "ョ"のフレーム
# 予測: argmax = 0 → 正しい！
```

### 4.4 Cross-Entropy Loss (補助情報)

```python
    # ステップ5: Cross-Entropy Loss
    
    # Reshape
    B, T, C = frame_disfluency_logits.shape  # (2, 75, 4)
    frame_logits_flat = frame_disfluency_logits.reshape(-1, C)  # (150, 4)
    frame_labels_flat = frame_level_labels.reshape(-1)          # (150,)
    
    # Cross-Entropy (ignore_index=-100 でblank除外)
    disfluency_loss = F.cross_entropy(
        frame_logits_flat,   # (150, 4)
        frame_labels_flat,   # (150,)
        ignore_index=-100,
        reduction='mean',
    )
    # disfluency_loss: スカラー (例: 0.35)
```

**具体的な損失計算**
```python
# サンプル0, フレーム25 (フィラー "エ" のフレーム)
logits = frame_logits_flat[25, :]  # [1.2, 8.5, 0.8, 1.1]
label = frame_labels_flat[25]       # 1 (filler)

probs = F.softmax(logits, dim=-1)   # [0.018, 0.963, 0.012, 0.007]
loss_t = -log(probs[1])             # -log(0.963) = 0.0377

# サンプル0, フレーム5 ("キ" のフレーム)
logits = frame_logits_flat[5, :]    # [8.2, 1.1, 0.6, 0.8]
label = frame_labels_flat[5]        # 0 (fluent)

probs = F.softmax(logits, dim=-1)   # [0.9985, 0.0008, 0.0003, 0.0004]
loss_t = -log(probs[0])             # -log(0.9985) = 0.0015

# サンプル0, フレーム30 (フィラー "ー" のフレーム)
logits = frame_logits_flat[30, :]   # [1.3, 8.7, 0.7, 1.0]
label = frame_labels_flat[30]       # 1 (filler)

probs = F.softmax(logits, dim=-1)   # [0.016, 0.968, 0.009, 0.007]
loss_t = -log(probs[1])             # -log(0.968) = 0.0325

# 全フレーム(ignoreを除く)の平均
# disfluency_loss ≈ (0.0377 + 0.0015 + 0.0325 + ...) / num_valid_frames
# 各文字のフレームごとに損失が計算され、平均される
```

### 4.5 総合損失

```python
    # ステップ6: 総合損失
    total_loss = asr_loss + self.disfluency_weight * disfluency_loss
    # total_loss = 2.45 + 1.0 * 0.35 = 2.80
    
    # 統計情報
    stats = {
        'loss': total_loss.detach(),
        'loss_asr': asr_loss.detach(),
        'loss_disfluency': disfluency_loss.detach(),
    }
    
    return total_loss, stats
```

---

## Phase 5: 推論時の出力フロー

### ファイル: `espnet2/asr/beam_search_transducer.py`

### 5.1 Beam Search (ASR)

```python
def beam_search(
    self,
    encoder_out: torch.Tensor,  # (1, T=75, 256) - 1発話
) -> List[Hypothesis]:
    """
    RNN-T Beam Search
    """
    
    beam_size = 10  # ビーム幅
    
    # 初期仮説
    hyps = [Hypothesis(score=0.0, yseq=[0], dec_state=None)]
    
    # 各フレームで展開
    for t in range(encoder_out.size(1)):  # T=75
        enc_out_t = encoder_out[:, t:t+1, :]  # (1, 1, 256)
        
        new_hyps = []
        for hyp in hyps:
            # Prediction Network
            y = torch.tensor([hyp.yseq[-1]])
            dec_out, new_state = self.decoder.score(y, hyp.dec_state, enc_out_t)
            # dec_out: (1, 1, 512)
            
            # Joint Network (ASRのみ)
            logits = self.joint_network(enc_out_t, dec_out)
            # logits: (1, 1, 1, 136) - ASR logits のみ
            
            log_probs = F.log_softmax(logits.squeeze(), dim=-1)
            # log_probs: (136,)
            
            # Top-k tokens
            top_k_scores, top_k_ids = torch.topk(log_probs, k=beam_size)
            
            for score, token_id in zip(top_k_scores, top_k_ids):
                if token_id == 0:  # blank
                    new_hyp = Hypothesis(
                        score=hyp.score + score.item(),
                        yseq=hyp.yseq,
                        dec_state=hyp.dec_state,
                    )
                else:  # non-blank
                    new_hyp = Hypothesis(
                        score=hyp.score + score.item(),
                        yseq=hyp.yseq + [token_id.item()],
                        dec_state=new_state,
                    )
                new_hyps.append(new_hyp)
        
        # Beam pruning
        hyps = sorted(new_hyps, key=lambda x: x.score, reverse=True)[:beam_size]
    
    return hyps
```

**Beam Search の可視化**
```
フレーム0:
  仮説0: [] score=0.0

フレーム1:
  仮説0: [] score=-0.5 (blank)
  仮説1: [23] score=-1.2 (キ)
  仮説2: [45] score=-2.1 (ョ)
  ...

フレーム5:
  仮説0: [23] score=-2.3 (キ)
  仮説1: [23, 45] score=-3.8 (キョ)
  仮説2: [] score=-4.1 (all blanks)
  ...

フレーム17:
  仮説0: [23, 45] score=-8.5 (キョ)
  仮説1: [23, 45, 12] score=-9.2 (キョウ) ← 今日
  仮説2: [23, 45, 12, 67] score=-10.5 (キョウワ) ← 今日は
  ...

フレーム35:
  仮説0: [23, 45, 12, 67, 8] score=-22.3 (キョウワエ)
  仮説1: [23, 45, 12, 67, 8, 9] score=-24.1 (キョウワエー) ← フィラー
  仮説2: [23, 45, 12, 67, 8, 9, 15] score=-26.8 (キョウワエート) ← 今日はえーと
  ...

最終 (フレーム74):
  Best: [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78] score=-85.3
  → "キョウワエートイイテンキデスネ" = "今日はえーと良い天気ですね"
```

### 5.2 補助情報検出 (推論時)

```python
def detect_disfluency_inference(
    self,
    encoder_out: torch.Tensor,  # (1, T=75, 256)
    best_hyp: Hypothesis,       # Best hypothesis from beam search
) -> List[int]:
    """
    推論時の補助情報検出
    """
    
    # ステップ1: yseq から labels を抽出 (blank除去)
    labels = [token for token in best_hyp.yseq if token != 0]
    # labels = [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78]
    # 15文字のID
    
    # ステップ2: Decoder で decoder_out を計算
    labels_tensor = torch.tensor([labels], device=encoder_out.device)
    decoder_out = self.decoder.batch_score(labels_tensor)
    # decoder_out: (1, U=15, 512)  # 15文字
    
    # ステップ3: Joint Network
    joint_outputs = self.joint_network(encoder_out, decoder_out)
    disfluency_logits = joint_outputs["disfluency_logits"]
    # disfluency_logits: (1, T=75, U=15, 4)
    
    # ステップ4: Character-level prediction
    # 方法1: 時間軸で平均
    char_level_logits = disfluency_logits.mean(dim=1)  # (1, U=15, 4)
    
    # 方法2: Viterbi alignmentを使って対応フレームの予測を集約
    # (より正確だが計算コストが高い)
    
    # ステップ5: Argmax で予測
    disfluency_preds = char_level_logits.argmax(dim=-1)  # (1, U=15)
    disfluency_labels = disfluency_preds.squeeze(0).tolist()
    # disfluency_labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #                     [キ,ョ,ウ,ワ,エ,ー,ト,イ,イ,テ,ン,キ,デ,ス,ネ]
    
    return disfluency_labels
```

**推論結果の例**
```python
# 入力音声: "今日はえーと良い天気ですね"
# データ形式: "キ ョ ウ ワ エ ー ト イ イ テ ン キ デ ス ネ" (カタカナ文字単位)

# ASR結果:
best_hyp.yseq = [23, 45, 12, 67, 8, 9, 15, 33, 33, 55, 67, 23, 89, 102, 78]
text = ["キ", "ョ", "ウ", "ワ", "エ", "ー", "ト", "イ", "イ", "テ", "ン", "キ", "デ", "ス", "ネ"]
#       今   日         は    え   ー  と   良   い   天   気         で   す  ね

# 補助情報検出結果（文字レベル）:
disfluency_labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#                    き ょ う は え ー と い い て ん き で す ね

# 統合結果:
文字  | 補助情報      | 単語構成
------|--------------|----------
キ    | fluent       | 今
ョ    | fluent       | 日
ウ    | fluent       | 
ワ    | fluent       | は
エ    | filled_pause | え
ー    | filled_pause | ー  ← フィラー検出
ト    | filled_pause | と
イ    | fluent       | 良
イ    | fluent       | い
テ    | fluent       | 天
ン    | fluent       | 
キ    | fluent       | 気
デ    | fluent       | で
ス    | fluent       | す
ネ    | fluent       | ね
```

---

## 出力データの完全な流れ (まとめ)

```
【入力】
encoder_out:  (B, T, 256)     # 音響表現
labels:       (B, U)           # 文字ID (U=15文字)
isdysfl:      (B, U)           # 補助情報ラベル (character-level, U=15)
ctc_logits:   (B, T, vocab)    # Viterbi用

↓ [Embedding]
decoder_in:   (B, U, 256)      # 文字埋め込み (U=15)

↓ [LSTM Decoder]
decoder_out:  (B, U, 512)      # 文脈表現 (U=15)
  ├─ decoder_out[b,0,:] = "文字0" までの文脈 (例: "キ")
  ├─ decoder_out[b,1,:] = "文字0,1" までの文脈 (例: "キョ")
  ├─ decoder_out[b,2,:] = "文字0,1,2" までの文脈 (例: "キョウ")
  └─ decoder_out[b,u,:] = "文字0〜u" までの文脈

↓ [Joint Network]
├─ encoder_out:  (B, T, 1, 256)
├─ decoder_out:  (B, 1, U, 512)  # U=15文字
└─ Broadcast + Add + Tanh

joint_out:    (B, T, U, 1024)  # 全組み合わせの統合表現 (T=75, U=15)
  └─ joint_out[b,t,u,:] = フレームt × 文字位置u の表現
  例: joint_out[b, 25, 4, :] = フレーム25で"エ"(位置4)を出力する表現

↓ [Output Heads]
├─ [ASR Head]
│  asr_logits: (B, T, U, vocab_size)  # (B, 75, 15, 136)
│    └─ asr_logits[b,t,u,v] = フレームt, 文字位置u で文字v のスコア
│
└─ [Disfluency Head]
   disfluency_logits: (B, T, U, 4)  # (B, 75, 15, 4)
     └─ disfluency_logits[b,t,u,c] = フレームt, 文字位置u でクラスc のスコア

↓ [損失計算]
├─ [RNN-T Loss]
│  input:  asr_logits (B, 75, 15, 136)
│  target: labels (B, 15)  # 文字単位
│  output: asr_loss (scalar)
│    └─ Forward-backward で全アライメントパスを計算
│       各文字が適切なフレームに割り当てられるよう学習
│
└─ [CE Loss + Viterbi]
   ├─ Viterbi: ctc_logits → frame_level_labels (B, T)
   │   文字レベルラベルをフレームレベルにマッピング
   │   例: "エ"(filler) → フレーム21-26が全てfiller
   ├─ Pooling: disfluency_logits → frame_logits (B, T, 4)
   │   15文字位置で平均pooling
   ├─ CE Loss: frame_logits vs frame_level_labels
   └─ output: disfluency_loss (scalar)

↓ [Total Loss]
total_loss = asr_loss + λ * disfluency_loss

↓ [Backward]
勾配計算: ∂loss/∂(全パラメータ)
  ├─ ∂loss/∂asr_output_weights
  ├─ ∂loss/∂disfluency_output_weights
  ├─ ∂loss/∂joint_weights
  ├─ ∂loss/∂decoder_weights
  ├─ ∂loss/∂encoder_weights
  └─ ∂loss/∂ctc_weights  ← Viterbi経由で学習

↓ [Update]
optimizer.step(): 全パラメータを更新

【推論時】
encoder_out → Beam Search → best_hypothesis
                             ↓
                           yseq = [23, 45, 12, 67, 8, 9, 15, ...]  # 15文字
                             ↓
                  ┌──────────┴──────────┐
                  ↓                     ↓
              [ASR結果]            [補助情報検出]
          "キョウワエート..."    [0, 0, 0, 0, 1, 1, 1, ...]
          (15文字)               (15文字レベルラベル)
```

---

## 重要なポイント

### 1. **3次元から4次元への拡張**
- Encoder: (B, T, 256) - フレームレベル (T=75フレーム)
- Decoder: (B, U, 512) - **文字レベル** (U=15文字)
- Joint: (B, T, U, 1024) - **全組み合わせ** (75×15=1125個の組み合わせ)

### 2. **文字単位の処理**
- トークン = 1文字（カタカナ）
- "キョウワエート" = 7文字が7トークン
- ASR、補助情報ともに文字レベルで処理

### 3. **Dual Task Learning**
- ASR: RNN-T Loss (文字単位 sequence-to-sequence)
- Disfluency: CE Loss (frame-level classification with character-level labels)

### 4. **Viterbi の役割**
- CTC logits で最適アライメント
- Character-level → Frame-level 変換
- 各文字のフレーム範囲を決定（例: "エ"→フレーム21-26）
- 補助情報の教師信号を全フレームに提供

### 5. **推論の2段階**
1. Beam Search で ASR 結果（文字単位）
2. Joint Network で補助情報検出

