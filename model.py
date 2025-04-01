# About Huggingface dataset:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/datasets_overview_pt.ipynb

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
  def __init__(self, d_model:int, vocab_size:int)->None:
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # (batch, seq_len) --> (batch, seq_len, d_model)
    # Multiply by sqrt(d_model) to scale the embeddings according to the paper
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model:int, seq_length:int, dropout:float)->None:
    super().__init__()
    self.seq_length = seq_length
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)

    # For even indices: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) # Where pos is the position and i is the dimension (the embedding dimension index)
    # For odd indices: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    # Create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_length, d_model) # self.pe not required - will be registered later

    # create a tensor of shape (seq_length, 1)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

    # -math.log(10000.0)/d_model: The negative sign is used because we want to divide by 10000^(2i/d_model), which is equivalent to multiplying by exp(-i/d_model * log(10000)).
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)) # (d_model / 2)


    # Apply sine to even position and cos to odd position
    pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
    pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

    # Add a batch dimension to the positional encoding
    pe = pe.unsqueeze(0) # (1, seq_len, d_model)
    # Register the positional encoding as a buffer
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) # (batch, seq_len, d_model)
    return self.dropout(x)

class LayerNormalization(nn.Module):
  def __init__(self, features: int, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter. For multiplication
    self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter. For Addition

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff) # w1 & b1; Bias is True by default
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model) # w2 and b2

  def forward(self, x):
    # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) # Note: d_ff=2048 as per paper
    x = self.linear1(x)
    x = torch.relu(x)  # Note: it's a functional relu often used in custom fwd fun. It is not torch.nn.relu which is used in model defination
    x = self.dropout(x)
    # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
    x = self.linear2(x)
    return x

class ResidualConnection(nn.Module):
  def __init__(self, features: int, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization(features)

  # sublayer is the previous module/layer
  def forward(self, x, sublayer):
    x = self.norm(x)
    x = sublayer(x)
    x = x + self.dropout(x)
    return x
    #return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float)->None:
    super().__init__()
    self.d_model = d_model # Embedding vector size, as per paper: d_model=512
    self.h = h # Number of heads
    assert d_model % h == 0, "d_model is not divisible by h"

    self.d_k = d_model // h # Dimension of vector seen by each head
    self.w_q = nn.Linear(d_model, d_model, bias=False) #Wq
    self.w_k = nn.Linear(d_model, d_model, bias=False) #Wk
    self.w_v = nn.Linear(d_model, d_model, bias=False) #Wv

    self.w_o = nn.Linear(d_model, d_model, bias=False) #Wo
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]

    # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) ==> (Batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)

    # before applying softmax we need to apply mask to hide the interaction among some tokens/word.
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = nn.functional.softmax(attention_scores, dim=-1) # (batch, h, seq_len, seq_len). Note: functional softmax

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # shape of `attention_scores @ value`: (batch, h, seq_len, d_k)
    return (attention_scores @ value), attention_scores # # return attention scores which can be used for visualization

  '''
  Note: Instead of passing q, k, v directly as arguments, we could have just passed the input x,
         and then computed query, key, value inside the function using linear projections. But, why
        do we pass q,k,v three args which is essentaially embedding in input_seq ?
        - Passing q,k,v provides flexibility to support both selfAttention & MultiHeadAttention
        - For implementing just selfAttention, we can use a single input in fwd function and calculates query, key, value
        - But, for cross-attention, we need to q from decoder input and k,v from encoder output.
  '''
  def forward(self, q, k, v, mask):
    query = self.w_q(q)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    key = self.w_k(k)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    value = self.w_v(v)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

    # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

    # Calculate attention by calling static method with class name
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # Combine all the heads together
    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k) # d_k = self.h*self.d_k

    # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    return self.w_o(x)

class EncoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    # self.residual_connection1 = ResidualConnection(dropout)
    # self.residual_connection2 = ResidualConnection(dropout)
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    #x = self.residual_connection1(x, self.self_attention_block(x, x, x, src_mask))
    #x = self.residual_connection2(x, self.feed_forward_block(x))
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # lambda means x is taken as input and used as parameter.
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x

class Encoder(nn.Module):
  # layers is nothing but stacked one or more EncoderBlock. Ex: Encoder(nn.ModuleList(encoder_blocks))
  def __init__(self,  features: int, layers: nn.ModuleList)-> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization(features)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class DecoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x

class Decoder(nn.Module):
  # layers is nothing but stacked one or more DecoderBlock. Ex: Decoder(nn.ModuleList(decoder_blocks))
  def __init__(self, features: int, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization(features)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x)

class ProjectionLayer(nn.Module):
  # The layers outside the Decoder Blocks are: linear and
  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x)->None:
    # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim=-1) # log_softmax is used for numerical stability suggested outside paper

class Transformer(nn.Module):
  '''
  - inputs->[Encoder]=> outpus: encoder_output, src_mask
  - outputs(right shifted) -> [Decoder]
  - Projection [liner->log_softmax]: probabilities of all tokens in vocab_size

  '''
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask) # Decoder's fwd function

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512, # as per paper
                      N: int = 6, # number stacked encoder/decoder layer
                      h: int = 8, # no. of heads as per paper
                      dropout: float = 0.1,
                      d_ff: int = 2048, # no. of hidden layers of FF
                      ) -> Transformer:

                      # create the embedding layers
                      src_embed = InputEmbeddings(d_model, src_vocab_size)
                      tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

                      # create positional encoding layers
                      src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
                      tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

                      ##############################
                      # create the encoder blocks  #
                      ##############################
                      encoder_blocks = []
                      for _ in range(N):
                        # each encoder block contains - one MultiHeadAttention, one FeedForwardBlock and 3) two ResidualConnection
                        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
                        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
                        # EncoderBlock class is used to makes an Encoder using ResidualConnection class
                        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
                        encoder_blocks.append(encoder_block)

                      #############################
                      # create the decoder blocks #
                      #############################
                      decoder_blocks = []
                      for _ in range(N):
                        # each decoder  block contains - two MultiHeadAttention, one FeedForwardBlock and three ResidualConnection
                        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
                        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
                        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
                        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
                        decoder_blocks.append(decoder_block)

                      # create the encoder and decoder
                      encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
                      decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

                      # create the projection layer
                      projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

                      # Create the transformer
                      transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

                      # initialize the params
                      for p in transformer.parameters():
                        if p.dim() > 1:
                          nn.init.xavier_uniform_(p)

                      return transformer
