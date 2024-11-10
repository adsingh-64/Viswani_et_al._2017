# Load WMT-14 French to English data
# ------------------------------------------------------------------------------
from datasets import load_dataset
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


dataset = load_dataset("wmt14", "fr-en", split="train[:1000000]")

# Model Global Variables
# ------------------------------------------------------------------------------
vocab_size = 50258
encoder_block_size = 30
decoder_block_size = 20
d_model = 64
n_head = 4
n_encoder_layers = 6
n_decoder_layers = 6
dropout = 0.2
batch_size = 32


# Data Preprocessing
# ------------------------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
eos_token_id = 50256
pad_token_id = 50257


def tokenize_function(examples):
    french_sentences = [item["fr"] for item in examples["translation"]]
    english_sentences = [item["en"] for item in examples["translation"]]
    french_tokens = [
        torch.tensor(tokenizer.encode(sentence)) for sentence in french_sentences
    ]
    english_tokens = [
        torch.tensor(tokenizer.encode(sentence)) for sentence in english_sentences
    ]
    return french_tokens, english_tokens


french_sentences, english_sentences = tokenize_function(dataset)


def pad_or_truncate(sequences, max_length, pad_token_id):
    padded_sequences = []
    for seq in sequences:
        seq_length = seq.shape[0]
        if seq_length > max_length:
            seq = seq[:max_length]
        else:
            padding_length = max_length - seq_length
            padding = torch.full((padding_length,), pad_token_id, dtype=torch.long)
            seq = torch.cat([seq, padding], dim=0)
        padded_sequences.append(seq)
    return torch.stack(padded_sequences)


encoder_inputs = pad_or_truncate(french_sentences, encoder_block_size, pad_token_id)

decoder_inputs = []
decoder_labels = []

for seq in english_sentences:
    seq_length = seq.shape[0]
    seq = seq[: decoder_block_size - 1]
    # Prepare decoder inputs by prepending <EOS>
    input_seq = torch.cat([torch.tensor([eos_token_id]), seq], dim=0)
    # Prepare decoder labels by appending <EOS>
    label_seq = torch.cat([seq, torch.tensor([eos_token_id])], dim=0)
    # Pad decoder inputs to decoder_max_context_length if needed
    if input_seq.shape[0] < decoder_block_size:
        padding_length = decoder_block_size - input_seq.shape[0]
        padding = torch.full((padding_length,), pad_token_id, dtype=torch.long)
        input_seq = torch.cat([input_seq, padding], dim=0)
    # Pad decoder labels to decoder_max_context_length if needed
    if label_seq.shape[0] < decoder_block_size:
        padding_length = decoder_block_size - label_seq.shape[0]
        padding = torch.full((padding_length,), pad_token_id, dtype=torch.long)
        label_seq = torch.cat([label_seq, padding], dim=0)
    decoder_inputs.append(input_seq)
    decoder_labels.append(label_seq)

decoder_inputs = torch.stack(decoder_inputs)
decoder_labels = torch.stack(decoder_labels)


def get_batch():
    ix = torch.randint(encoder_inputs.shape[0], (batch_size,))
    enc_inputs = encoder_inputs[ix].to(device)  # [batch_size, encoder_context_length]
    dec_inputs = decoder_inputs[ix].to(device)  # [B, decoder_context_length]
    dec_labels = decoder_labels[ix].to(device)  # [B, decoder_context_length]
    return enc_inputs, dec_inputs, dec_labels


# Feedforward Sublayer
# ------------------------------------------------------------------------------
class FeedForward(nn.Module):
    """linear layer followed by non-linearity"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # [B, T, n_embd]
            nn.ReLU(),
            nn.Linear(
                4 * d_model, d_model
            ),  # transition matrix to prepare for going back into residual pathway via addition
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Encoder
# ------------------------------------------------------------------------------
class EncoderMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(d_model, 2)  # all [B, T, C]
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        full_att = q @ k.transpose(-2, -1) / (k.shape[-1]) ** 0.5  # [B, nh, T, T]
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        full_att = full_att.masked_fill(mask == 0, float("-inf"))
        attention_scores = F.softmax(full_att, dim=-1)
        context_vectors = attention_scores @ v  # [B, nh, T, hs]
        context_vectors = (
            context_vectors.transpose(1, 2).contiguous().view(B, T, C)
        )  # concat heads -- [B, T, C]
        out = self.dropout(self.proj(context_vectors))
        return out


class Encoder_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = EncoderMultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.sa(
            self.ln1(x), mask
        )  # departure from Attention is All You Need -- we apply LN before transformation
        x = x + self.ffwd(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.token_embedding_table = shared_embedding
        self.position_embedding_table = nn.Embedding(encoder_block_size, d_model)
        self.blocks = nn.ModuleList([Encoder_Block() for _ in range(n_encoder_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        mask = (idx != pad_token_id).to(idx.device)  # [B, T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return x


# Decoder
# ------------------------------------------------------------------------------
class DecoderMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "tril", torch.tril(torch.ones(decoder_block_size, decoder_block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(d_model, 2)  # all [B, T, C]
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # [B, nh, T, hs]
        full_att = q @ k.transpose(-2, -1) / (k.shape[-1]) ** 0.5  # [B, nh, T, T]
        left_att = full_att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention_scores = F.softmax(left_att, dim=-1)
        context_vectors = attention_scores @ v  # [B, nh, T, hs]
        context_vectors = (
            context_vectors.transpose(1, 2).contiguous().view(B, T, C)
        )  # concat heads -- [B, T, C]
        out = self.dropout(self.proj(context_vectors))
        return out


class DecoderEncoderMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.q = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, encoder_outputs_mask):
        B, T_decoder, C = x.shape
        B, T_encoder, C = encoder_outputs.shape
        kv = self.kv(encoder_outputs)  # [B, T_encoder, 2 * d_model]
        k, v = kv.split(d_model, 2)  # each [B, T_encoder, d_model]
        q = self.q(x)  # [B, T_decoder, d_model]
        q = q.view(B, T_decoder, n_head, C // n_head).transpose(
            1, 2
        )  # [B, nh, T_decoder, hs]
        k = k.view(B, T_encoder, n_head, C // n_head).transpose(
            1, 2
        )  # [B, nh, T_encoder, hs]
        v = v.view(B, T_encoder, n_head, C // n_head).transpose(
            1, 2
        )  # [B, nh, T_encoder, hs]
        full_att = (
            q @ k.transpose(-2, -1) / (k.shape[-1]) ** 0.5
        )  # [B, nh, T_decoder, T_encoder]
        mask = encoder_outputs_mask.unsqueeze(1).unsqueeze(2)
        full_att = full_att.masked_fill(mask == 0, float("-inf"))
        attention_scores = F.softmax(full_att, dim=-1)
        context_vectors = attention_scores @ v  # [B, nh, T_decoder, hs]
        context_vectors = (
            context_vectors.transpose(1, 2).contiguous().view(B, T_decoder, C)
        )  # concat heads -- [B, T_decoder, d_model]
        out = self.dropout(self.proj(context_vectors))
        return out


class Decoder_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = DecoderMultiHeadAttention()
        self.sa2 = DecoderEncoderMultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_outputs, encoder_outputs_mask):
        x = x + self.sa1(
            self.ln1(x)
        )  # departure from Attention is All You Need -- we apply LN before transformation
        x = x + self.ffwd(self.ln2(x))
        x = x + self.sa2(self.ln3(x), encoder_outputs, encoder_outputs_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.token_embedding_table = shared_embedding
        self.position_embedding_table = nn.Embedding(decoder_block_size, d_model)
        self.blocks = nn.ModuleList([Decoder_Block() for _ in range(n_decoder_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(
            d_model, vocab_size
        )  # need to tie to the input embedding table (transpose of it)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, encoder_outputs, encoder_outputs_mask, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (batch_size, block_size, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (block_size, n_embd)
        x = tok_emb + pos_emb  #  (batch_size, block_size, n_embd)
        for block in self.blocks:
            x = block(x, encoder_outputs, encoder_outputs_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, block_size, vocab_size)
        if targets is None:
            loss = None
        else:
            # idx and targets are both (B,T) tensor of integers
            B, T, _ = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=50257)
        return logits, loss


# Full Model
# ------------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            self.shared_embedding.weight /= d_model**0.5
        self.encoder = Encoder(self.shared_embedding)
        self.decoder = Decoder(self.shared_embedding)

    def forward(self, encoder_inputs, decoder_inputs, targets=None):
        encoder_outputs_mask = (encoder_inputs != pad_token_id).to(device)
        encoder_outputs = self.encoder(encoder_inputs)
        logits, loss = self.decoder(
            decoder_inputs, encoder_outputs, encoder_outputs_mask, targets
        )
        return logits, loss


# Training
# ------------------------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Device: {device}")

torch.set_float32_matmul_precision("high")

transformer = Transformer().to(device)
transformer = torch.compile(transformer)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=6e-4)
max_iters = 50000
eval_iter = 2000

num_batches = dataset.num_rows // batch_size
num_epochs = max_iters // num_batches
print(f"Number of epochs: {num_epochs}")

for iter in range(max_iters):
    enc_inputs, dec_inputs, dec_labels = get_batch()
    logits, loss = transformer(enc_inputs, dec_inputs, dec_labels)
    if iter % eval_iter == 0:
        print(f"Step {iter} | Loss: {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Translation
# ------------------------------------------------------------------------------
transformer.eval()


class Hypothesis:
    def __init__(self, tokens, log_prob, length):
        self.tokens = tokens
        self.log_prob = log_prob
        self.length = length

    def get_score(self, alpha=0.6):
        return self.log_prob / (self.length**alpha)


sentence = "Bonjour tout le monde!"
encoder_inputs = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
encoder_outputs_mask = (encoder_inputs != pad_token_id).to(device)
encoder_outputs = transformer.encoder(encoder_inputs)

k = 4

initial_hypothesis = Hypothesis([eos_token_id], 0.0, 1)
beam = [initial_hypothesis]
completed_hypotheses = []

for _ in range(
    decoder_block_size - 1
):  # predefined cutoff, when it runs the decoder_block_sizeth time, the decoder input will be decoder_block_size length
    new_beam = []
    for hypothesis in beam:
        decoder_input = torch.tensor(
            [hypothesis.tokens], dtype=torch.long, device=device
        )

        with torch.no_grad():
            logits, _ = transformer.decoder(
                decoder_input, encoder_outputs, encoder_outputs_mask, targets=None
            )

        next_token_logits = logits[:, -1, :].view(-1)
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        top_k_log_probs, top_k_indices = torch.topk(next_token_log_probs, k)  # both [k]
        next_tokens = top_k_indices.tolist()
        next_token_log_probs = top_k_log_probs.tolist()

        for token, log_prob in zip(next_tokens, next_token_log_probs):
            new_tokens = hypothesis.tokens + [token]
            new_log_prob = hypothesis.log_prob + log_prob
            new_length = hypothesis.length + 1
            new_hypothesis = Hypothesis(new_tokens, new_log_prob, new_length)

            if token == eos_token_id or len(new_tokens) > decoder_block_size:
                completed_hypotheses.append(new_hypothesis)
            else:
                new_beam.append(new_hypothesis)

    if not new_beam:  # stop early if all hypotheses have been completed
        break

    beam = sorted(new_beam, key=lambda h: h.get_score(), reverse=True)[:k]  # keep top k

if completed_hypotheses:
    best_hypothesis = max(completed_hypotheses, key=lambda h: h.get_score())
else:  # if not completed hypotheses, then beam must be non-empty so if-else partitions event space
    best_hypothesis = max(beam, key=lambda h: h.get_score())

output_tokens = best_hypothesis.tokens[1:]

translation = tokenizer.decode(output_tokens, errors="replace")

print(f"Translation: {translation}")

"""
ubuntu@150-136-145-43:~$ python NMT.py
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 104.16it/s]
Device: cuda
Number of epochs: 1
Step 0 | Loss: 11.367871284484863
Step 2000 | Loss: 5.373536109924316
Step 4000 | Loss: 5.069731712341309
Step 6000 | Loss: 4.477414131164551
Step 8000 | Loss: 4.165456295013428
Step 10000 | Loss: 4.095539093017578
Step 12000 | Loss: 4.2008819580078125
Step 14000 | Loss: 3.9759016036987305
Step 16000 | Loss: 3.8721559047698975
Step 18000 | Loss: 3.9698171615600586
Step 20000 | Loss: 3.2516841888427734
Step 22000 | Loss: 3.7683522701263428
Step 24000 | Loss: 3.5908992290496826
Step 26000 | Loss: 3.961156129837036
Step 28000 | Loss: 3.124617099761963
Step 30000 | Loss: 2.917598009109497
Step 32000 | Loss: 3.261178493499756
Step 34000 | Loss: 3.234287977218628
Step 36000 | Loss: 3.0169413089752197
Step 38000 | Loss: 3.559629440307617
Step 40000 | Loss: 2.851454734802246
Step 42000 | Loss: 3.3791704177856445
Step 44000 | Loss: 2.797959804534912
Step 46000 | Loss: 3.1998775005340576
Step 48000 | Loss: 3.030996799468994
Translation: Congratulations to the world!<|endoftext|> 
:((
"""
