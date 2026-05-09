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

# 追加課題

一通り動いたら、以下を順番に追加する。

---

## 追加課題1: temperature sampling

`generate()` に `temperature` を追加する。

```python
logits = logits / temperature
```

---

## 追加課題2: top-k sampling

確率上位 k 個の token だけから sampling する。

---

## 追加課題3: validation loss

train/val split を作り、validation loss を定期的に計算する。

---

## 追加課題4: 学習曲線の保存

loss を list に保存して、あとで plot できるようにする。

---

## 追加課題5: KV cache なし generate の計算量を確認する

通常の `generate()` では、毎 step で過去 token 全体を再計算していることを確認する。

---

## 追加課題6: KV cache あり generate を実装する

各 layer の K, V を保存して、decode 時に新しい token の Q, K, V だけを計算する。

---

## 追加課題7: RoPE に変更する

learned positional embedding をやめて、Rotary Positional Embedding を使う。

---

## 追加課題8: LayerNorm を RMSNorm に変更する

LLaMA 系の構成に近づける。

---

## 追加課題9: MLP を SwiGLU に変更する

通常の MLP を SwiGLU に置き換える。

---

## 追加課題10: Grouped Query Attention を実装する

Multi-head Attention を GQA に変更する。

---

# 最終的に理解したいこと

この課題を通じて、以下を説明できる状態を目指す。

- なぜ token を整数 ID にするのか
- なぜ次 token 予測では target を1つ右にずらすのか
- Q, K, V の役割
- causal mask の役割
- multi-head attention の shape
- residual connection の意味
- LayerNorm の位置
- logits と probability の違い
- `CrossEntropyLoss` に渡す shape
- autoregressive generation の流れ
- KV cache が何を保存しているのか
- decode 時になぜ memory-bound になりやすいのか