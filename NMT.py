# Load WMT-14 French to English data
# ------------------------------------------------------------------------------
from datasets import load_dataset
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

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

encoder_inputs = encoder_inputs.to(device)
decoder_inputs = decoder_inputs.to(device)
decoder_labels = decoder_labels.to(device)

transformer = Transformer().to(device)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=6e-4)
max_iters = 50000
eval_iter = 2000

for iter in range(max_iters):
    enc_inputs, dec_inputs, dec_labels = get_batch()
    logits, loss = transformer(enc_inputs, dec_inputs, dec_labels)
    if iter % eval_iter == 0:
        print(f"Loss: {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Translation
# ------------------------------------------------------------------------------
transformer.eval()

@torch.no_grad
def translate_sentence(transformer, sentence, max_length=decoder_block_size - 1):
    encoder_input = (torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device))  # Shape: [1, seq_len]
    encoder_outputs_mask = (encoder_input != pad_token_id).to(device)
    encoder_outputs = transformer.encoder(encoder_input)

    decoder_input = torch.tensor([[eos_token_id]], dtype=torch.long, device=device)  # Shape: [1, 1]

    # Start decoding loop
    for _ in range(max_length):
        logits, _ = transformer.decoder(decoder_input, encoder_outputs, encoder_outputs_mask, targets=None)

        # Get the next token (greedy decoding)
        next_token_logits = logits[:, -1, :]  # Get logits for the last time step
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Shape: [1, 1]

        # Append the predicted token to the decoder input
        decoder_input = torch.cat([decoder_input, next_token_id], dim=1)  # Shape: [1, current_seq_len + 1]

        # Check if the <EOS> token is generated
        if next_token_id.item() == eos_token_id:
            break

    # Remove the initial <EOS> token from the decoder input since this is the translation
    output_tokens = decoder_input[0, 1:].tolist()

    # Decode the tokens to English text
    translation = tokenizer.decode(output_tokens, errors="replace")
    return translation


french_sentence = "Bonjour tout le monde!"

# Translate the sentence
translation = translate_sentence(transformer, french_sentence)
print(f"Translation: {translation}")

"""
ubuntu@158-101-125-195:~$ python NMT.py
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10.5k/10.5k [00:00<00:00, 94.5MB/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 52.12it/s]
train-00000-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 252M/252M [00:01<00:00, 236MB/s]
train-00001-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 241M/241M [00:01<00:00, 232MB/s]
train-00002-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 243M/243M [00:01<00:00, 240MB/s]
train-00003-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 247M/247M [00:01<00:00, 229MB/s]
train-00004-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242M/242M [00:01<00:00, 240MB/s]
train-00005-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 238M/238M [00:00<00:00, 240MB/s]
train-00006-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240M/240M [00:01<00:00, 230MB/s]
train-00007-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 241M/241M [00:01<00:00, 228MB/s]
train-00008-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242M/242M [00:01<00:00, 197MB/s]
train-00009-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 239M/239M [00:01<00:00, 235MB/s]
train-00010-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 239M/239M [00:01<00:00, 237MB/s]
train-00011-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 241M/241M [00:01<00:00, 239MB/s]
train-00012-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 241M/241M [00:01<00:00, 239MB/s]
train-00013-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230M/230M [00:00<00:00, 235MB/s]
train-00014-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214M/214M [00:00<00:00, 222MB/s]
train-00015-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231M/231M [00:01<00:00, 225MB/s]
train-00016-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 227M/227M [00:00<00:00, 241MB/s]
train-00017-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 226M/226M [00:00<00:00, 239MB/s]
train-00018-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 261M/261M [00:01<00:00, 203MB/s]
train-00019-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 259M/259M [00:01<00:00, 245MB/s]
train-00020-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 261M/261M [00:01<00:00, 246MB/s]
train-00021-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 264M/264M [00:01<00:00, 216MB/s]
train-00022-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 267M/267M [00:01<00:00, 239MB/s]
train-00023-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270M/270M [00:01<00:00, 241MB/s]
train-00024-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 274M/274M [00:01<00:00, 243MB/s]
train-00025-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 278M/278M [00:01<00:00, 237MB/s]
train-00026-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 365M/365M [00:01<00:00, 211MB/s]
train-00027-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 322M/322M [00:01<00:00, 221MB/s]
train-00028-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 370M/370M [00:01<00:00, 237MB/s]
train-00029-of-00030.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 311M/311M [00:01<00:00, 241MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:37<00:00,  1.24s/files]
validation-00000-of-00001.parquet: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 475k/475k [00:00<00:00, 91.4MB/s]
test-00000-of-00001.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 536k/536k [00:00<00:00, 256MB/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40836715/40836715 [00:59<00:00, 688954.32 examples/s]
Generating validation split: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 586643.29 examples/s]
Generating test split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3003/3003 [00:00<00:00, 553721.15 examples/s]
Device: cuda
Loss: 11.365097999572754
Loss: 5.1328277587890625
Loss: 4.8307647705078125
Loss: 4.530695915222168
Loss: 4.498836517333984
Loss: 4.260554313659668
Loss: 3.619072675704956
Loss: 3.9896106719970703
Loss: 3.842914342880249
Loss: 3.85141921043396
Loss: 3.908470869064331
Loss: 3.7891368865966797
Loss: 3.407797336578369
Loss: 3.505903482437134
Loss: 3.477874755859375
Loss: 3.3102025985717773
Loss: 3.395603895187378
Loss: 3.4466094970703125
Loss: 3.0760834217071533
Loss: 3.256739377975464
Loss: 2.8746440410614014
Loss: 2.780557155609131
Loss: 2.8517420291900635
Loss: 3.1370038986206055
Loss: 2.9124841690063477
Translation: Many of all the world.<|endoftext|> 
:(
"""
