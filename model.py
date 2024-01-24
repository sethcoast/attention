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
        self.att_mask = self.create_look_ahead_mask()
    
    def forward(self, x):
        x = self.mhAtt(x)
        x = self.dropout(x)
        
        return x

    def create_look_ahead_mask(size):
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
    def __init__(self, dmodel, numLayers = 6):
        super(TransformerNet, self).__init__()
        # todo: define the positional encoding stuff?

        # Instantiate encoder/decoder layers (EncoderLayer is employed inside the DecoderLayer class)
        self.layers = nn.ModuleList()
        for _ in range(numLayers):
            self.layers.append(DecoderLayer(dmodel))
        
        # todo: do I need an extra output layer (review the paper, you might)
    
    # todo: maybe rewrite this eventually
    def forward(self, x):
        # split x into input/output sequences
        input, output = x[0], x[1]
        # Embedding layers (multiply weights by √dmodel)

        # Contatenate w/ positional encoding (optional)

        # Feed through the network
        for layer in self.layers:
            x = layer(in_emb, out_emb)

        return x