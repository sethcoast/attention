# Import the shit
import torch
import torch.nn as nn
import torch.utils.data


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
    def __init__(self, dmodel, mask=False):
        super(MultiheadAttentionBlock, self).__init__()
        self.mhAtt = nn.MultiheadAttention(dmodel, 8)
        self.dropout = nn.Dropout(0.1)
        # look-ahead mask
        # self.att_mask = self.create_look_ahead_mask()
    
    def forward(self, x):
        x = self.mhAtt(x)
        x = self.dropout(x)
        
        return x

    def create_look_ahead_mask(self, size):
        # todo: double check that this is correct
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
# define the encoder model
class EncoderLayer(nn.Module):
    def __init__(self, dmodel):
        super(EncoderLayer, self).__init__()
        # Define the attention layer
        self.mhaBlock = MultiheadAttentionBlock(dmodel)
        self.ff = FeedForwardBlock(dmodel)
        
    
    def forward(self, x):
        # Multi Head Attention Sublayer()
        out = self.mhaBlock(x)
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
        self.encoder = EncoderLayer(dmodel)
        self.mhaBlock1 = MultiheadAttentionBlock(dmodel, mask=True)
        self.ffBlock = FeedForwardBlock(dmodel)
        self.mhaBlock2 = MultiheadAttentionBlock(dmodel)
    
    def forward(self, in_encoding, out_embedding):
        ## Multi Head Attention Sublayer 1 (masked)
        # todo: padding mask?
        decoded = self.mhaBlock1(out_embedding, out_embedding, out_embedding)
        decoded += out_embedding # residual connection
        decoded = self.norm(decoded) # normalization
        ## Multi Head Attention Sublayer 2
        # todo: do I need to concatenate or do something else?

        ## Feed Forward Sublayer


class TransformerNet(nn.Module):
    # numLayers = 6 by default to mimic the parameters of the paper
    def __init__(self, vocab_size, dmodel, numLayers = 6):
        super(TransformerNet, self).__init__()
        # todo: define the positional encoding stuff?
        # Embedding layer
        self.embedding_layer = nn.Linear(vocab_size, dmodel)
        # Instantiate encoder/decoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(numLayers):
            self.encoder_layers.append(EncoderLayer(dmodel))
        self.decoder_layers = nn.ModuleList()
        for _ in range(numLayers):
            self.decoder_layers.append(DecoderLayer(dmodel))
        # Output layer
        self.last = nn.Linear(dmodel, vocab_size)
        self.last.weight = self.embedding_layer.weight # todo: double check this
        self.softmax = nn.Softmax(dim=vocab_size)
        # todo: do I need an extra output layer (review the paper, you might)
    
    # todo: maybe rewrite this eventually
    def forward(self, in_seq, out_seq):
        # todo: Embedding layers (multiply weights by √dmodel (v2))
        # todo: Contatenate input w/ positional encoding (v2)
        # Encoder stack
        in_encode = self.embedding_layer(in_seq) # shape = (batch_size, seq_len, dmodel)
        for layer in self.encoder_layers:
            in_encode = layer(in_encode)
        # Decoder stack
        out_encode = self.embedding_layer(out_seq)
        for layer in self.decoder_layers:
            out_encode = layer(in_encode, out_encode)
        # Output layer
        out = self.last(out_encode)
        out = self.softmax(out)

        return out