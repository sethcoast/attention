# Import the shit
import torch
import torch.nn as nn
import torch.utils.data
import math

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    return mask  # Shape: [size, size]

def create_padding_mask(seq, pad_token=0):
    mask = (seq == pad_token).type(torch.bool)
    return mask  # Shape: [batch_size, seq_len]


# Feed forward block (sublayer)
class FeedForwardBlock(nn.Module):
    def __init__(self, dmodel):
        super(FeedForwardBlock, self).__init__()
        # Define the feed forward network layers
        self.fc1 = nn.Linear(dmodel, dmodel)
        self.fc2 = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True) # inplace = True to save memory

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

# Multihead Attention block (Sublayer)
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dmodel):
        super(MultiheadAttentionBlock, self).__init__()
        self.mhAtt = nn.MultiheadAttention(dmodel, 8, dropout=0.1, batch_first=True)

    def forward(self, Q, K, V, padding_mask=None, look_ahead_mask=None):
        # print(Q.shape, K.shape, V.shape)
        x = self.mhAtt(query=Q, key=K, value=V, need_weights=False, key_padding_mask=padding_mask, attn_mask=look_ahead_mask)
        return x[0]


# define the encoder model
class EncoderLayer(nn.Module):
    def __init__(self, dmodel):
        super(EncoderLayer, self).__init__()
        # Define the attention layer
        self.mhaBlock = MultiheadAttentionBlock(dmodel)
        self.ff = FeedForwardBlock(dmodel)
        self.norm = nn.LayerNorm(dmodel)


    def forward(self, x, padding_mask=None):
        # Multi Head Attention Sublayer()
        out = self.mhaBlock(x, x, x, padding_mask=padding_mask)
        out += x # residual connection
        out = self.norm(out) # normalization

        # Feed Forward Sublayer()
        out2 = self.ff(out)
        out2 += out # residual connection
        out2 = self.norm(out2) # normalization

        return out2

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, dmodel):
        super(DecoderLayer, self).__init__()
        self.mhaBlock1 = MultiheadAttentionBlock(dmodel)
        self.ffBlock = FeedForwardBlock(dmodel)
        self.mhaBlock2 = MultiheadAttentionBlock(dmodel)
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, src_encoding, target_embedding, src_padding_mask=None, target_padding_mask=None, look_ahead_mask=None):
        ## Multi Head Attention Sublayer 1 (masked)
        # todo: padding mask?
        decoded = self.mhaBlock1(target_embedding, target_embedding, target_embedding, padding_mask=target_padding_mask, look_ahead_mask=look_ahead_mask)
        decoded += target_embedding # residual connection
        decoded = self.norm(decoded) # normalization
        ## Multi Head Attention Sublayer 2
        decoded2 = self.mhaBlock2(decoded, src_encoding, src_encoding, padding_mask=src_padding_mask)
        decoded2 += decoded # residual connection
        decoded2 = self.norm(decoded2) # normalization
        ## Feed Forward Sublayer
        decoded3 = self.ffBlock(decoded2)
        decoded3 += decoded2 # residual connection
        decoded3 = self.norm(decoded3) # normalization

        return decoded3

# PositionalEncoding module provided by ChatGPT
class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough `position_encoding` to use during training.
        pe = torch.zeros(max_len, dmodel)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Not a parameter, just data, but we want it in the state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerNet(nn.Module):
    # numLayers = 6 by default to mimic the parameters of the paper
    def __init__(self, vocab_size, dmodel, seq_len, numLayers = 6):
        super(TransformerNet, self).__init__()
        self.dmodel = dmodel
        self.seq_len = seq_len
        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(dmodel)
        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, dmodel)
        # Instantiate encoder/decoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(numLayers):
            self.encoder_layers.append(EncoderLayer(dmodel))
        self.decoder_layers = nn.ModuleList()
        for _ in range(numLayers):
            self.decoder_layers.append(DecoderLayer(dmodel))


    # todo: maybe rewrite this eventually
    def forward(self, src_seq, target_seq, src_padding_mask=None, target_padding_mask=None, look_ahead_mask=None):
        # print("src_seq.shape: ", src_seq.shape)
        ## ENCODER STACK
        # Embedding layer
        src_encode = self.embedding_layer(src_seq) # shape = (batch_size, seq_len, dmodel)
        # Multiply weights by √dmodel
        src_encode = src_encode * math.sqrt(self.dmodel)
        # Positional encoding
        src_encode = self.pos_encoding(src_encode) # shape = (batch_size, seq_len, dmodel)
        for layer in self.encoder_layers:
            src_encode = layer(src_encode, padding_mask=src_padding_mask)
        # print("in_encode.shape: ", in_encode.shape)

        ## DECODER STACK
        # Embedding layer
        target_encode = self.embedding_layer(target_seq)
        # Multiply weights by √dmodel
        target_encode = target_encode * math.sqrt(self.dmodel)
        # Positional encoding
        target_encode = self.pos_encoding(target_encode)
        for layer in self.decoder_layers:
            target_encode = layer(src_encode, target_encode, src_padding_mask=src_padding_mask, target_padding_mask=target_padding_mask, look_ahead_mask=look_ahead_mask)
        # print("target_encode.shape: ", target_encode.shape)
        # Output layer: Reuse embedding weights
        out = torch.matmul(target_encode, self.embedding_layer.weight.transpose(0, 1))

        return out

    # greedy evaluation
    def translate(self, src_seq, start_token, end_token):
        self.eval()  # Set the model to evaluation mode
        
        # Ensure src_seq is a tensor and has the correct shape
        if not isinstance(src_seq, torch.Tensor):
            src_seq = torch.tensor(src_seq)
        src_seq = src_seq.unsqueeze(0)  # Add batch dimension if it's not there
        
        # Initialize target_seq with start_token
        target_seq = torch.tensor([[start_token]], dtype=torch.long)
        
        # Device handling (to support models on GPU)
        device = next(self.parameters()).device
        src_seq = src_seq.to(device)
        target_seq = target_seq.to(device)
        # Create attention masks
        src_padding_mask = create_padding_mask(src_seq).to(device)
        
        outputs = []
        for _ in range(self.seq_len):
            # Forward pass: obtain logits for the next token
            logits = self.forward(src_seq, target_seq, src_padding_mask)
            
            # Select the last token from the sequence
            # Use logits directly since they're used in the forward pass for computing softmax
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(-1).unsqueeze(-1)
            
            # Check if the next token is the end_token
            if next_token_id.item() == end_token:
                outputs.append(next_token_id.item())
                break
            
            # Append next_token_id to target_seq for generating next token
            target_seq = torch.cat([target_seq, next_token_id], dim=-1)
            outputs.append(next_token_id.item())

        return outputs