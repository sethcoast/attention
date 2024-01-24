# Import the shit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchtext
import torchvision
import torchvision.transforms
import json

from model import TransformerNet
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import numpy as np
# import matplotlib.pyplot as plt

# ok I really need to prepare this fucking data first. Ok so here's my list of unanswered questions:
# 1. How do I pad?
    # do I pad to the length of the longest sequence? (seems easy enough, if not a little memory inefficient)
# 2. How do I mask within a batch? Is there a way to do like a custom mask for each sequence?

class SentenceDataset(Dataset):
    def __init__(self, data_path):
        # read the data
        with open(data_path, 'r') as file:
            self.data = json.loads(json.load(file))
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return the input and output sequences
        return self.data[idx][0], self.data[idx][1]

def custom_collate_fn(batch):
    # Sort the batch in the descending order of sequence length
    batch.sort(key=lambda x: len(x), reverse=True)

    # Separate the sequences and labels (if any) # todo: this seems unecessary, remove
    sequences = [item[0] for item in batch]

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences

if __name__ == "__main__":
    # Data loading 
    # sort sentences by length (todo: do this in prepare_data.py when you write the sentences to file)
    # apply transforms as necessary (pad here?)
    data_dir = 'data/stage/'
    split_suffix = 'test'
    train_data = SentenceDataset(data_dir + split_suffix + '_bpe.json')

    # Move pytorch dataset into dataloader.
    train_batch_size = 10
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    print(f'Created `train_loader` with {len(train_loader)} batches!')

    # todo: create validation and test dataloaders

    # BucketIterator for batching todo: maybe remove this
    # train_loader = BucketIterator(train_data, batch_size=train_batch_size,
    #                                             sort_key=lambda x: len(x[0]),
    #                                             sort=False,
    #                                             sort_within_batch=True,
    #                                             shuffle=True)

    # Model, Loss Function, Optimizer
    dmodel = 512
    H = 6
    model = TransformerNet(dmodel, H)
    criterion = nn.CrossEntropyLoss()  # todo: I'm pretty sure this is correct, but double check
    optimizer = optim.Adam(model.parameters(), lr=0.001) # todo: double check paper params

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training Loop
    num_epochs = 5  # Example number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_idx, (input_seq, output) in enumerate(train_loader):
            input_seq, output_seq = input_seq.to(device), output_seq.to(device)

            # Forward pass
            outputs = model(input_seq, output_seq)
            loss = criterion(outputs, output_seq)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()  # Backpropagation
            optimizer.step()  # Apply the gradients

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training Complete")