# Mini Transformer 実装課題

このプロジェクトでは、文字レベルの Decoder-only Transformer、つまり小さな GPT 風モデルを 1 から実装する。

目的は、Transformer の以下の要素を自分で実装して理解することである。

- tokenization
- next token prediction
- causal self-attention
- multi-head attention
- residual connection
- feed-forward network
- Transformer block
- language modeling loss
- autoregressive generation

---

## 最終目標

以下が通る状態を目指す。

```bash
uv run pytest -q
```

その後、余力があれば以下も実行できるようにする。

```bash
uv run python src/train.py
uv run python src/generate.py
```

---

## 想定ディレクトリ構成

```text
Transformer-imp/
├── src/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── attention.py
│   ├── blocks.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── test/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_attention.py
│   ├── test_blocks.py
│   └── test_model.py
├── data/
│   └── input.txt
├── pyproject.toml
└── TASKS.md
```

---

## pytest 設定

`pyproject.toml` に以下を追加する。

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["test"]
python_files = ["test_*.py"]
```

また、`src` を package として import できるように、以下の空ファイルを作る。

```text
src/__init__.py
```

---

# 課題1: CharTokenizer を実装する

## ファイル

```text
src/tokenizer.py
```

## 目的

文字列を token ID に変換し、token ID から文字列に戻せるようにする。

今回は BPE や SentencePiece は使わず、**1文字 = 1 token** とする。

---

## 公開API

以下のクラスを実装する。

```python
class CharTokenizer:
    def __init__(self, text: str):
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...
```

---

## 仕様

`CharTokenizer(text)` は、`text` に含まれる文字を語彙として扱う。

```python
tokenizer = CharTokenizer("hello world")
```

`encode(text)` は文字列を整数 ID のリストに変換する。

```python
ids = tokenizer.encode("hello")
```

`decode(ids)` は整数 ID のリストを文字列に戻す。

```python
decoded = tokenizer.decode(ids)
```

このとき、以下が成り立つ必要がある。

```python
tokenizer.decode(tokenizer.encode("hello")) == "hello"
```

`vocab_size` は、初期化に使った `text` に含まれる一意な文字数を返す。

```python
tokenizer.vocab_size == len(set(text))
```

---

## 注意点

同じ文字は常に同じ ID に変換されること。

```python
tokenizer = CharTokenizer("banana")
ids = tokenizer.encode("banana")

ids[1] == ids[3] == ids[5]  # "a"
ids[2] == ids[4]            # "n"
```

初期化時の語彙に存在しない文字を `encode()` しようとした場合は、`KeyError` が発生してよい。

```python
tokenizer = CharTokenizer("hello")
tokenizer.encode("hello!")  # KeyError
```

---

## 内部実装について

内部で辞書を使う実装が自然だが、属性名は問わない。

例えば、以下のような対応表を持つと実装しやすい。

```python
# 文字 -> ID
{"h": 0, "e": 1, "l": 2, "o": 3}

# ID -> 文字
{0: "h", 1: "e", 2: "l", 3: "o"}
```

ただし、テストは内部属性名を要求しない。  
`encode`, `decode`, `vocab_size` の振る舞いだけを満たせばよい。

---

## テスト

```bash
uv run pytest test/test_tokenizer.py -q
```

---

## 合格条件

- `vocab_size` が一意文字数と一致する
- `encode()` が `list[int]` を返す
- `encode()` の出力長が入力文字列長と一致する
- `decode(encode(text)) == text` になる
- 同じ文字は同じ ID になる
- 未知文字で `KeyError` が発生する

---

# 課題2: CharDataset を実装する

## ファイル

```text
src/dataset.py
```

## 目的

長い token ID 列から、次 token 予測用の入力 `x` と正解 `y` を作る。

---

## 公開API

```python
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

---

## 仕様

例えば、以下の token ID 列があるとする。

```python
token_ids = [0, 1, 2, 3, 4, 5]
block_size = 4
```

このとき、`dataset[0]` は以下を返す。

```python
x = tensor([0, 1, 2, 3])
y = tensor([1, 2, 3, 4])
```

つまり、

```text
x: idx から block_size 個
y: idx + 1 から block_size 個
```

である。

`dataset[2]` なら以下になる。

```python
x = tensor([2, 3, 4, 5])
y = tensor([3, 4, 5, 6])
```

ただしこの例が成立するには、`token_ids` に `6` まで含まれている必要がある。

---

## 入出力

`__getitem__` は `(x, y)` を返す。

```python
x, y = dataset[idx]
```

shape はどちらも以下。

```text
(block_size,)
```

dtype はどちらも以下。

```text
torch.long
```

---

## length

`len(dataset)` は以下とする。

```python
len(token_ids) - block_size
```

これは、`y` を1 token 未来にずらして作るためである。

---

## テスト

```bash
uv run pytest test/test_dataset.py -q
```

---

## 合格条件

- `len(dataset) == len(token_ids) - block_size`
- `dataset[idx]` が `(x, y)` を返す
- `x` と `y` が `torch.Tensor`
- `x.shape == (block_size,)`
- `y.shape == (block_size,)`
- `x` と `y` が1 tokenずれている
- `x.dtype == torch.long`
- `y.dtype == torch.long`

---

# 課題3: CausalSelfAttention を実装する

## ファイル

```text
src/attention.py
```

## 目的

Transformer の中核である causal self-attention を実装する。

GPT型モデルでは、各 token は自分自身と過去 token だけを見る。  
未来 token は見てはいけない。

---

## 公開API

```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        dropout: float,
    ):
        ...

    def forward(self, x):
        ...
```

---

## 入出力

入力:

```text
x: (B, T, d_model)
```

出力:

```text
out: (B, T, d_model)
```

入力と同じ shape の tensor を返すこと。

---

## 仕様

`d_model` は `num_heads` で割り切れる必要がある。  
割り切れない場合は `AssertionError` を出す。

```python
assert d_model % num_heads == 0
```

`forward(x)` は、causal self-attention を計算する。

各位置 `t` の出力は、入力の位置 `0, 1, ..., t` のみに依存してよい。  
位置 `t+1` 以降の未来 token に依存してはいけない。

---

## 内部処理の考え方

典型的には以下の流れで実装する。

```text
x
↓
Linear projection
↓
q, k, v
↓
multi-head に分割
↓
scaled dot-product attention
↓
causal mask
↓
softmax
↓
V と掛ける
↓
head を結合
↓
output projection
```

---

## 重要な shape

入力:

```text
x: (B, T, d_model)
```

Q, K, V 作成後:

```text
q, k, v: (B, T, d_model)
```

head に分割後:

```text
q, k, v: (B, num_heads, T, head_dim)
```

attention score:

```text
att: (B, num_heads, T, T)
```

出力:

```text
out: (B, T, d_model)
```

---

## causal mask

未来 token を見ないように、下三角行列を使う。

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

0 の位置は `-inf` にしてから softmax するのが一般的。

---

## 内部実装について

以下のような実装詳細は問わない。

- `qkv_proj` という名前の layer を使うか
- Q, K, V を別々の Linear で作るか
- causal mask を `register_buffer` で持つか
- mask を forward 内で毎回作るか
- dropout layer の属性名

テストは、`CausalSelfAttention.forward(x)` の外部仕様だけを確認する。

---

## テスト

```bash
uv run pytest test/test_attention.py -q
```

---

## 合格条件

- 出力 shape が入力 shape と一致する
- `d_model` が `num_heads` で割り切れないとき `AssertionError`
- 出力が finite である
- backward が通る
- 入力 `x` に gradient が流れる
- 未来 token を変更しても、それより前の出力が変わらない

---

# 課題4: FeedForward と TransformerBlock を実装する

## ファイル

```text
src/blocks.py
```

## 目的

Attention と MLP を組み合わせて Transformer Block を作る。

---

## 公開API

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        ...

    def forward(self, x):
        ...


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        dropout: float,
    ):
        ...

    def forward(self, x):
        ...
```

---

## FeedForward の仕様

入力と同じ shape の tensor を返す。

入力:

```text
x: (B, T, d_model)
```

出力:

```text
out: (B, T, d_model)
```

典型的には以下の構造を使う。

```text
Linear(d_model, 4 * d_model)
GELU
Linear(4 * d_model, d_model)
Dropout
```

ただし、テストは内部 layer 名や exact な構成までは確認しない。  
公開APIと入出力仕様を満たせばよい。

---

## TransformerBlock の仕様

入力と同じ shape の tensor を返す。

入力:

```text
x: (B, T, d_model)
```

出力:

```text
out: (B, T, d_model)
```

推奨構造は Pre-LN Transformer。

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

ただし、テストは内部属性名や Pre-LN/Post-LN の違いまでは直接確認しない。  
最小課題としては、shape が保たれ、backward が通ればよい。

---

## テスト

```bash
uv run pytest test/test_blocks.py -q
```

---

## 合格条件

- `FeedForward` の出力 shape が入力 shape と一致する
- `FeedForward` で backward が通る
- `TransformerBlock` の出力 shape が入力 shape と一致する
- `TransformerBlock` の出力が finite である
- `TransformerBlock` で backward が通る
- 入力 `x` に gradient が流れる

---

# 課題5: MiniGPT を実装する

## ファイル

```text
src/model.py
```

## 目的

Decoder-only Transformer 全体を実装する。

---

## 公開API

```python
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        ...

    def forward(self, input_ids, targets=None):
        ...

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int):
        ...
```

---

## モデル構造

典型的には以下の構造にする。

```text
input_ids
↓
Token Embedding
↓
Position Embedding
↓
TransformerBlock × num_layers
↓
LayerNorm
↓
Linear Head
↓
logits
```

---

## forward の入力

```text
input_ids: (B, T)
targets:   (B, T) or None
```

`input_ids` は token ID の tensor で、dtype は `torch.long` を想定する。

---

## forward の出力

```python
logits, loss = model(input_ids, targets=None)
```

`logits`:

```text
(B, T, vocab_size)
```

`loss`:

```text
targets が None の場合: None
targets が与えられた場合: scalar tensor
```

---

## sequence length 制約

`T` が `block_size` を超える場合は `AssertionError` を出す。

```python
assert T <= self.block_size
```

---

## loss の仕様

`targets` が与えられた場合、各位置で次 token を予測する cross entropy loss を計算する。

`logits` は以下の shape。

```text
(B, T, vocab_size)
```

`targets` は以下の shape。

```text
(B, T)
```

`F.cross_entropy` に渡すために flatten するのが一般的。

```python
loss = F.cross_entropy(
    logits.view(B * T, vocab_size),
    targets.view(B * T),
)
```

---

## generate の仕様

`generate(input_ids, max_new_tokens)` は、入力 token 列に `max_new_tokens` 個の token を追加した tensor を返す。

入力:

```text
input_ids: (B, T)
```

出力:

```text
out: (B, T + max_new_tokens)
```

生成された token ID は、以下の範囲に入っている必要がある。

```text
0 <= token_id < vocab_size
```

---

## generate の典型的な流れ

```text
input_ids
↓
最後の block_size token だけ取り出す
↓
model に入れる
↓
最後の位置の logits を取り出す
↓
次 token を選ぶ
↓
input_ids に結合
↓
繰り返す
```

次 token の選び方は問わない。

例えば以下のどちらでもよい。

- argmax
- multinomial sampling

テストは sampling 方法を要求しない。

---

## テスト

```bash
uv run pytest test/test_model.py -q
```

---

## 合格条件

- `targets=None` のとき、`logits` と `None` を返す
- `logits.shape == (B, T, vocab_size)`
- `targets` があるとき、scalar loss を返す
- loss が finite である
- backward が通る
- 少なくとも一部の model parameter に gradient が流れる
- `T > block_size` のとき `AssertionError`
- `generate()` の出力長が `max_new_tokens` 分増える
- `generate()` の出力 dtype が `torch.long`
- 生成 token ID が vocab 範囲内にある

---

# 課題6: train.py を実装する

## ファイル

```text
src/train.py
```

## 目的

`data/input.txt` を使って MiniGPT を学習する。

この課題は pytest の対象ではなく、実際に学習が回ることを手動で確認する。

---

## 実装する処理

```text
1. data/input.txt を読む
2. CharTokenizer を作る
3. text を token ID に変換する
4. CharDataset を作る
5. DataLoader を作る
6. MiniGPT を作る
7. optimizer を作る
8. training loop を回す
9. checkpoint を保存する
```

---

## 最小 config 例

```python
data_path = "data/input.txt"
block_size = 128
batch_size = 32
d_model = 128
num_heads = 4
num_layers = 4
dropout = 0.1
lr = 3e-4
max_steps = 5000
save_path = "checkpoint.pt"
```

---

## checkpoint に保存するとよいもの

```python
{
    "model": model.state_dict(),
    "tokenizer_state": ...,
    "config": {
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
    }
}
```

`tokenizer_state` の中身は実装に合わせて自由に決めてよい。  
ただし、`generate.py` で tokenizer を復元できる情報を保存すること。

---

## 実行

```bash
uv run python src/train.py
```

---

## 合格条件

- loss が表示される
- step が進む
- `checkpoint.pt` が保存される
- 学習中に shape error が出ない

---

# 課題7: generate.py を実装する

## ファイル

```text
src/generate.py
```

## 目的

学習済み checkpoint から文章を生成する。

この課題も pytest の対象ではなく、実際に生成できることを手動で確認する。

---

## 実装する処理

```text
1. checkpoint.pt を読む
2. tokenizer を復元する
3. MiniGPT を作る
4. model.load_state_dict() する
5. prompt を encode する
6. model.generate() を呼ぶ
7. decode して表示する
```

---

## 実行

```bash
uv run python src/generate.py
```

---

## 合格条件

- `checkpoint.pt` を読み込める
- prompt を token ID に変換できる
- `generate()` が動く
- 出力を文字列に decode できる
- 生成テキストが表示される

---

# 課題8: utils.py を実装する

## ファイル

```text
src/utils.py
```

## 目的

補助関数をまとめる。

この課題は必須ではない。  
必要になったタイミングで実装してよい。

---

## 実装する関数例

```python
def set_seed(seed: int):
    ...

def get_device():
    ...

def count_parameters(model):
    ...
```

---

## set_seed

乱数 seed を固定する。

```python
set_seed(42)
```

内部で以下を設定するのが一般的。

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

---

## get_device

CUDA が使えるなら `"cuda"`、そうでなければ `"cpu"` を返す。

```python
device = get_device()
```

---

## count_parameters

学習可能パラメータ数を返す。

```python
num_params = count_parameters(model)
```

---

# 実行順

まず tokenizer から順に通す。

```bash
uv run pytest test/test_tokenizer.py -q
```

次に dataset。

```bash
uv run pytest test/test_dataset.py -q
```

次に attention。

```bash
uv run pytest test/test_attention.py -q
```

次に block。

```bash
uv run pytest test/test_blocks.py -q
```

最後に model。

```bash
uv run pytest test/test_model.py -q
```

全部まとめて実行する場合。

```bash
uv run pytest -q
```

---

# 学習のチェックポイント

## tokenizer.py

文字列が整数列になることを理解する。

```text
"hello" → [3, 1, 0, 0, 2]
```

ID の具体的な値は実装によって変わってよい。

---

## dataset.py

次 token 予測の教師データを理解する。

```text
input:  [h, e, l, l]
target: [e, l, l, o]
```

---

## attention.py

各 token が過去 token だけを見ることを理解する。

```text
token 0 → token 0 のみ
token 1 → token 0, 1
token 2 → token 0, 1, 2
token 3 → token 0, 1, 2, 3
```

---

## blocks.py

Attention と MLP を residual connection で積むことを理解する。

```python
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

---

## model.py

token ID 列から、各位置の次 token 分布を出すことを理解する。

```text
input_ids: (B, T)
logits:    (B, T, vocab_size)
```

---

## train.py

`logits` と `targets` から cross entropy loss を計算することを理解する。

---

## generate.py

最後の位置の logits から次 token を生成することを理解する。

```text
prompt
↓
next token
↓
append
↓
next token
↓
append
...
```

---

# 追加課題: LLM 推論高速化ロードマップ

ここから先は、学習よりも **推論** に焦点を当てる。

目的は、Decoder-only Transformer における以下を理解することである。

- Autoregressive decoding
- Prefill / Decode
- KV cache
- compute-bound / memory-bound
- batching
- continuous batching
- PagedAttention / vLLM
- FlashAttention
- MQA / GQA
- quantization
- speculative decoding
- prefix caching
- long-context inference

---

# 追加課題1: Autoregressive decoding を理解する

## 目的

Decoder-only Transformer が、1 token ずつ次 token を生成する仕組みを理解する。

---

## 学ぶこと

通常の `generate()` では、以下を繰り返す。

```text
input_ids
↓
model(input_ids)
↓
最後の位置の logits を取り出す
↓
次 token を選ぶ
↓
input_ids に append
↓
繰り返し
```

---

## 実装課題

`MiniGPT.generate()` にコメントを追加し、各行が何をしているか説明する。

特に以下を明確にする。

```python
input_cond = input_ids[:, -self.block_size:]
logits, _ = self(input_cond)
logits = logits[:, -1, :]
next_id = ...
input_ids = torch.cat([input_ids, next_id], dim=1)
```

---

## 理解チェック

以下を説明できること。

- なぜ最後の位置の logits だけを使うのか
- なぜ生成は1 tokenずつ進むのか
- なぜ `input_ids` が毎 step 長くなるのか
- なぜ `block_size` より長い文脈は切り詰めるのか

---

# 追加課題2: Prefill / Decode を理解する

## 目的

LLM 推論が大きく2段階に分かれることを理解する。

```text
Prefill phase
Decode phase
```

---

## Prefill

prompt 全体を一度にモデルへ通す段階。

```text
prompt tokens: (B, T_prompt)
↓
Transformer forward
↓
各 layer の K, V を作る
↓
最後の token から次 token を予測する
```

Prefill では、prompt 内の全 token に対して attention を計算する。

---

## Decode

生成済み token に続けて、1 token ずつ新しい token を生成する段階。

```text
new token: (B, 1)
↓
Transformer forward
↓
過去の KV cache を参照
↓
次 token を予測する
```

---

## 実装課題

まずは実装せず、`generate()` の中で以下をコメントとして分ける。

```python
# Prefill 相当:
# 最初の prompt を model に入れて、最初の next token を出す

# Decode 相当:
# 以降は、生成された token を追加しながら1 tokenずつ進める
```

その後、`generate_naive()` を作る。

```python
def generate_naive(self, input_ids, max_new_tokens: int):
    ...
```

これは KV cache を使わず、毎 step で全 token を再計算する実装にする。

---

## 理解チェック

以下を説明できること。

- Prefill は何を計算しているのか
- Decode は何を計算しているのか
- なぜ Prefill は並列化しやすいのか
- なぜ Decode は1 tokenずつになりやすいのか
- なぜ KV cache が Decode で重要になるのか

---

# 追加課題3: KV cache を実装する

## 目的

Decode 時に、過去 token の K, V を再計算せず保存して使う仕組みを理解する。

---

## 背景

通常の naive decoding では、毎 step で全 token をモデルに入れる。

```text
step 1: [t1, t2, t3]
step 2: [t1, t2, t3, t4]
step 3: [t1, t2, t3, t4, t5]
```

この場合、過去 token の K, V を何度も再計算してしまう。

KV cache では、過去 token の K, V を保存する。

```text
past K, V
+
new token の K, V
↓
attention
```

---

## 実装課題

`CausalSelfAttention.forward()` を拡張する。

```python
def forward(self, x, past_kv=None, use_cache=False):
    ...
    return out, new_kv
```

`past_kv` は以下の形を想定する。

```python
past_kv = (past_k, past_v)
```

shape は以下。

```text
past_k: (B, num_heads, T_past, head_dim)
past_v: (B, num_heads, T_past, head_dim)
```

新しく計算した `k`, `v` を結合する。

```python
if past_kv is not None:
    past_k, past_v = past_kv
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
```

---

## 注意

KV cache ありの decode では、入力 `x` は基本的に1 token。

```text
x: (B, 1, d_model)
```

しかし attention では、query は1 token、key/value は過去全体になる。

```text
q: (B, num_heads, 1, head_dim)
k: (B, num_heads, T_total, head_dim)
v: (B, num_heads, T_total, head_dim)
```

attention score は以下。

```text
att: (B, num_heads, 1, T_total)
```

---

## 理解チェック

以下を説明できること。

- KV cache は何を保存しているのか
- Q は保存しないのか
- Decode 時の Q, K, V の shape
- KV cache ありとなしで、何の再計算が減るのか
- KV cache を使っても attention 自体は過去 token 全体を見る必要があること

---

# 追加課題4: KV cache あり generate を実装する

## 目的

`generate_naive()` と `generate_with_cache()` の違いを実装で理解する。

---

## 実装課題

`MiniGPT` に以下を追加する。

```python
@torch.no_grad()
def generate_with_cache(self, input_ids, max_new_tokens: int):
    ...
```

流れは以下。

```text
1. prompt を prefill する
2. 各 layer の KV cache を保存する
3. 最後の token から next token を生成する
4. 以降は new token だけを model に入れる
5. KV cache を更新しながら decode する
```

---

## 必要な変更

`TransformerBlock.forward()` も cache を受け取れるようにする。

```python
def forward(self, x, past_kv=None, use_cache=False):
    ...
    return x, new_kv
```

`MiniGPT.forward()` も layer ごとの cache を受け取れるようにする。

```python
def forward(self, input_ids, targets=None, past_kvs=None, use_cache=False):
    ...
    return logits, loss, new_kvs
```

---

## 理解チェック

以下を比較する。

```python
out1 = model.generate_naive(input_ids, max_new_tokens=20)
out2 = model.generate_with_cache(input_ids, max_new_tokens=20)
```

sampling を `argmax` に固定すれば、両者の出力が一致することを確認しやすい。

---

# 追加課題5: 推論ボトルネック compute-bound vs memory-bound を理解する

## 目的

Prefill と Decode でボトルネックが変わる理由を理解する。

---

## Prefill

Prefill は、多数の token をまとめて処理する。

```text
(B, T_prompt, d_model)
```

行列積が大きく、GPU の演算器を使いやすい。

そのため、比較的 compute-bound になりやすい。

---

## Decode

Decode は、基本的に1 tokenずつ処理する。

```text
(B, 1, d_model)
```

一方で、各 layer で過去の KV cache を大量に読む必要がある。

そのため、比較的 memory-bound になりやすい。

---

## 実装課題

`generate_naive()` と `generate_with_cache()` の時間を測る。

```python
import time

start = time.time()
model.generate_naive(...)
print(time.time() - start)

start = time.time()
model.generate_with_cache(...)
print(time.time() - start)
```

小さいモデルでは差が見えにくいが、以下を変えて傾向を見る。

```python
block_size
num_layers
d_model
num_heads
max_new_tokens
```

---

## 理解チェック

以下を説明できること。

- compute-bound とは何か
- memory-bound とは何か
- なぜ Prefill は compute-bound になりやすいのか
- なぜ Decode は memory-bound になりやすいのか
- KV cache は計算量を減らすが、メモリ読み出し量を増やす側面があること

---

# 追加課題6: Batching を理解する

## 目的

複数リクエストをまとめて推論する batching を理解する。

---

## 学ぶこと

batching では、複数の prompt をまとめてモデルに入れる。

```text
request 1: "Hello"
request 2: "The cat"
request 3: "Once upon"
```

これらを padding して batch にする。

```text
input_ids: (B, T)
```

---

## 実装課題

`generate_batch()` を作る。

```python
def generate_batch(self, input_ids, attention_mask, max_new_tokens):
    ...
```

最低限、同じ長さの prompt だけを batch として扱ってよい。

---

## 理解チェック

以下を説明できること。

- なぜ batching で throughput が上がるのか
- batching しても latency が悪化する場合がある理由
- prompt 長が違うと padding が必要になる理由
- padding token を attention から除外する必要がある理由

---

# 追加課題7: Continuous batching を理解する

## 目的

LLM サーバーで、リクエストを逐次追加・削除しながら batch を維持する仕組みを理解する。

---

## 背景

普通の batching では、batch 内の全 request が終わるまで次の request を入れにくい。

continuous batching では、

```text
終わった request を batch から抜く
新しい request を batch に入れる
```

を decode step ごとに行う。

---

## 実装課題

本格実装はしない。  
代わりに、擬似コードを書く。

```python
active_requests = []

while True:
    add_new_requests(active_requests)
    run_one_decode_step(active_requests)
    remove_finished_requests(active_requests)
```

---

## 理解チェック

以下を説明できること。

- static batching と continuous batching の違い
- なぜ LLM serving では continuous batching が重要なのか
- decode step 単位で batch を組み替える意味
- request ごとに生成長が違うと何が困るのか

---

# 追加課題8: PagedAttention / vLLM を理解する

## 目的

KV cache を効率的に管理する PagedAttention の考え方を理解する。

---

## 背景

長い文脈や多数 request を扱うと、KV cache が巨大になる。

単純に連続メモリとして確保すると、以下が問題になる。

- メモリ断片化
- 余分な予約領域
- request ごとの長さの違い
- batch の入れ替わり

PagedAttention は、KV cache を固定長 block に分けて管理する。

---

## 学ぶイメージ

通常の KV cache:

```text
request A: [token token token token token token]
request B: [token token]
```

PagedAttention:

```text
request A: [block 1] -> [block 2] -> [block 3]
request B: [block 4]
```

request ごとに論理的には連続しているように見せつつ、物理メモリでは page/block 単位で管理する。

---

## 実装課題

本格実装はしない。  
代わりに、簡易的な KV block manager を Python で書く。

```python
class KVBlockManager:
    def allocate(self, request_id, num_tokens):
        ...

    def free(self, request_id):
        ...

    def append_token(self, request_id):
        ...
```

---

## 理解チェック

以下を説明できること。

- なぜ KV cache 管理が難しいのか
- PagedAttention は何を page/block 化しているのか
- vLLM が continuous batching と相性がよい理由
- メモリ断片化をどう減らすのか

---

# 追加課題9: FlashAttention を理解する

## 目的

attention のメモリ効率を改善する FlashAttention の考え方を理解する。

---

## 背景

通常の attention では、attention score を明示的に作る。

```text
att: (B, num_heads, T, T)
```

T が長いと、この行列が巨大になる。

FlashAttention は、巨大な attention 行列を GPU HBM に保存せず、block-wise に計算することでメモリ使用量を減らす。

---

## 注意

FlashAttention は主に attention 計算そのものの効率化であり、KV cache とは別の話。

ただし、long-context inference ではどちらも重要になる。

---

## 実装課題

本格実装はしない。  
以下を比較して理解する。

```python
torch.nn.functional.scaled_dot_product_attention
```

PyTorch の SDPA を使う版と、自前 attention 版の置き換えを試す。

---

## 理解チェック

以下を説明できること。

- 通常 attention がなぜ `T x T` メモリを使うのか
- FlashAttention が何を保存しないのか
- FlashAttention と KV cache の違い
- Prefill で FlashAttention が効きやすい理由

---

# 追加課題10: MQA / GQA を理解する

## 目的

Decode 時の KV cache サイズを減らす MQA / GQA を理解する。

---

## 通常の Multi-Head Attention

Q, K, V がすべて `num_heads` 個ある。

```text
Q: num_heads
K: num_heads
V: num_heads
```

---

## MQA

Multi-Query Attention では、Q は複数 head だが、K, V は1組だけ。

```text
Q: num_heads
K: 1 head
V: 1 head
```

KV cache が小さくなる。

---

## GQA

Grouped-Query Attention では、複数の Q head が同じ K, V head を共有する。

```text
Q heads: 8
KV heads: 2
```

この場合、4個の Q head が1個の KV head を共有する。

---

## 実装課題

`CausalSelfAttention` に `num_kv_heads` を追加する。

```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        block_size,
        dropout,
    ):
        ...
```

通常 MHA:

```python
num_kv_heads = num_heads
```

MQA:

```python
num_kv_heads = 1
```

GQA:

```python
num_kv_heads < num_heads
```

---

## 理解チェック

以下を説明できること。

- MHA, MQA, GQA の違い
- KV cache サイズがどう変わるか
- Decode が memory-bound なとき、なぜ MQA/GQA が効くのか
- 共有された K, V を Q head 側にどう broadcast するのか

---

# 追加課題11: Quantization を理解する

## 目的

重みや KV cache を低精度化して、メモリ使用量と帯域を減らす考え方を理解する。

---

## 学ぶこと

代表的には以下がある。

- weight-only quantization
- activation quantization
- KV cache quantization

---

## 実装課題

本格的な int8 kernel は実装しない。  
代わりに、重み tensor を擬似的に int8 quantize / dequantize する関数を書く。

```python
def quantize_int8(x):
    ...

def dequantize_int8(q, scale):
    ...
```

---

## 理解チェック

以下を説明できること。

- なぜ quantization でメモリ使用量が減るのか
- なぜ Decode の memory-bound 改善に効くのか
- weight quantization と KV cache quantization の違い
- 精度劣化が起きる理由

---

# 追加課題12: Speculative decoding を理解する

## 目的

小さい draft model と大きい target model を使って、生成を高速化する考え方を理解する。

---

## 背景

通常の decoding は1 tokenずつ進む。

speculative decoding では、小さいモデルが複数 token を先に提案する。

```text
draft model: token を k 個提案
target model: まとめて検証
```

受理された token は一気に進める。

---

## 実装課題

本格的な確率的検証は難しいため、まずは toy version を作る。

```python
def speculative_decode_toy(draft_model, target_model, input_ids, k):
    ...
```

最初は以下の単純化でよい。

- draft model が k token 生成する
- target model も同じ位置で argmax を出す
- 一致した token だけ受理する

---

## 理解チェック

以下を説明できること。

- なぜ speculative decoding は速くなりうるのか
- どのような場合に効果が小さいのか
- draft model と target model の役割
- target model による検証が必要な理由

---

# 追加課題13: Prefix caching を理解する

## 目的

同じ prefix を持つ複数 request で、prefill 計算を再利用する考え方を理解する。

---

## 背景

チャットや RAG では、複数 request が同じ system prompt や長い共通文脈を持つことがある。

```text
request A: [system prompt] + user question A
request B: [system prompt] + user question B
```

このとき、`system prompt` 部分の KV cache を再利用できる。

---

## 実装課題

toy 実装として、prefix token IDs を key にして cache する。

```python
prefix_cache = {}

key = tuple(prefix_ids)

if key in prefix_cache:
    past_kvs = prefix_cache[key]
else:
    _, _, past_kvs = model(prefix_ids, use_cache=True)
    prefix_cache[key] = past_kvs
```

---

## 理解チェック

以下を説明できること。

- prefix caching は何を再利用しているのか
- Prefill のどの部分が省略できるのか
- なぜ system prompt が長いと効果が大きいのか
- KV cache を使い回すときの注意点

---

# 追加課題14: Long-context inference を理解する

## 目的

長い context を扱うとき、計算量・メモリ量・品質に何が起きるか理解する。

---

## 学ぶこと

長い context では、主に以下が問題になる。

```text
Prefill:
attention 計算が重い

Decode:
KV cache が巨大になる

Serving:
request ごとの KV cache 管理が難しい
```

---

## 実装課題

`block_size` を大きくしたときのメモリ使用量と速度を観察する。

```python
block_size = 128
block_size = 512
block_size = 1024
```

以下を見る。

- prefill 時間
- decode 時間
- KV cache の tensor shape
- GPU memory usage

---

## 理解チェック

以下を説明できること。

- context length が長くなると Prefill はなぜ重くなるのか
- context length が長くなると Decode はなぜ KV cache 読み出しが重くなるのか
- FlashAttention が効く部分
- PagedAttention が効く部分
- MQA/GQA が効く部分
- prefix caching が効く部分

---

# 推奨学習順

以下の順番で進める。

```text
1. Autoregressive decoding
2. Prefill / Decode
3. KV cache
4. KV cache あり generate
5. compute-bound vs memory-bound
6. Batching
7. Continuous batching
8. PagedAttention / vLLM
9. FlashAttention
10. MQA / GQA
11. Quantization
12. Speculative decoding
13. Prefix caching
14. Long-context inference
```

---

# このロードマップの中心

この追加課題では、常に以下の問いに戻る。

```text
LLM inference では、どの tensor をいつ計算して、
どの tensor を保存し、
どの tensor を何度も読むのか？
```

特に重要なのは以下。

```text
Prefill:
prompt 全体を並列に処理して KV cache を作る

Decode:
新しい token を1つずつ処理し、過去の KV cache を読む

KV cache:
各 layer の K, V を保存する

Decode bottleneck:
計算そのものより、KV cache の読み出しが支配的になりやすい
```