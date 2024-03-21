# Import the shit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchtext
import torchvision
import torchvision.transforms
from tqdm import tqdm
from transformers import AutoTokenizer
import json

from model import TransformerNet
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import numpy as np
# import matplotlib.pyplot as plt

class EnJpTranslationDataset(Dataset):
    def __init__(self, en_sentences, jp_sentences, tokenizer, max_length=128):
        self.en_sentences = en_sentences
        self.jp_sentences = jp_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        # Get the English and Japanese sentences
        en_s = self.en_sentences[idx]
        jp_s = self.jp_sentences[idx]

        # Encode the English sentence (source)
        source_encoded = self.tokenizer.encode_plus(
            en_s,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode the Japanese sentence (target)
        # Note: Depending on your model, you might need to add special tokens manually
        target_encoded = self.tokenizer.encode_plus(
            jp_s,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return source and target as a dictionary
        # Flatten the tensors to remove unnecessary batch dimension added by return_tensors='pt'
        return {
            'source_input_ids': source_encoded['input_ids'].squeeze(0),
            'source_attention_mask': source_encoded['attention_mask'].squeeze(0),
            'target_input_ids': target_encoded['input_ids'].squeeze(0),
            'target_attention_mask': target_encoded['attention_mask'].squeeze(0),
        }

if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load a pre-trained multilingual tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    # Data loading
    drive_dir = 'drive/MyDrive/'
    data_dir = 'drive/MyDrive/data/ja-en/split/'
    file_path = data_dir + 'test' # todo: change this when you want to use a different dataset
    # split english and japanese sentences into separate lists
    en_sentences, jp_sentences = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            en, ja = line.strip().split('\t')
            en_sentences.append(en)
            jp_sentences.append(ja)

    # create a dataset
    train_data = EnJpTranslationDataset(en_sentences, jp_sentences, tokenizer)
    # Move pytorch dataset into dataloader.
    train_batch_size = 32
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
    seq_len = train_data.max_length
    vocab_size = tokenizer.vocab_size
    model = TransformerNet(vocab_size, dmodel, seq_len, H)
    criterion = nn.CrossEntropyLoss()  # todo: I'm pretty sure this is correct, but double check
    optimizer = optim.Adam(model.parameters(), lr=0.001) # todo: double check paper params

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)

    # Training Loop
    train_losses = []
    val_losses = []
    num_epochs = 1  # Example number of epochs
    for epoch in range(num_epochs):
        loop = tqdm(total=len(train_loader), position=0, leave=False)
        model.train()  # Set the model to training mode
        for batch_idx, train_dict in enumerate(train_loader):
            src_seq = train_dict['source_input_ids']
            target_seq = train_dict['target_input_ids']

            src_seq, target_seq = src_seq.to(device), target_seq.to(device)
            # Create attention masks
            # src_padding_mask = create_padding_mask(src_seq, pad_token_id).to(device)
            # target_padding_mask = create_padding_mask(target_seq, pad_token_id).to(device)
            # look_ahead_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
            src_padding_mask = None
            target_padding_mask = None
            look_ahead_mask = None

            print("src_seq")
            print(src_seq.shape)
            print(src_seq)
            print("target_seq")
            print(target_seq.shape)
            print(target_seq)
            # print("src_padding_mask")
            # print(src_padding_mask.shape)
            # print(src_padding_mask)
            # print("target_padding_mask")
            # print(target_padding_mask.shape)
            # print(target_padding_mask)
            # print("look_ahead_mask")
            # print(look_ahead_mask.shape)
            # print(look_ahead_mask)

            # Forward pass
            outputs = model(src_seq, target_seq, src_padding_mask, target_padding_mask, look_ahead_mask)
            # print("outputs")
            # print(outputs.shape)
            # print(outputs)
            # print(target_seq.shape)
            # print(target_seq)
            # print(outputs[0][0].shape)
            # print(outputs[0][0].sum())
            # break
            preds = torch.argmax(outputs, dim=-1)
            print("preds")
            print(preds.shape)
            print(preds)

            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, target_seq)
            print("loss")
            print(loss)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()  # Backpropagation
            optimizer.step()  # Apply the gradients


            # if (batch_idx + 1) % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Housekeeping
            train_losses.append(loss)
            # accuracy = get_raw_accuracy(y_hat, y_truth)
            mem = torch.cuda.memory_allocated(0) / 1e9
            # accuracies.append(accuracy)
            # loop.set_description('epoch:{}, loss:{:.4f}, accuracy:{:.3f}, mem:{:.2f}'.format(epoch, loss, accuracy, mem))
            loop.set_description('epoch:{}, loss:{:.4f}, mem:{:.2f}'.format(epoch, loss, mem))
            loop.update(1)
            torch.cuda.empty_cache()
            break

        # Validation Loop (cute and optimized)
        # compute the loss for all x, y in val_loader, then get the mean of those losses
        # val = np.mean([criterion(model(x.cuda()), y.cuda().long()).item()
        #                 for x, y in test_loader
        #                 ])
        # val_losses.append((len(train_losses), val))
        # print('\nVal Loss: {:.4f}'.format(val))

        # todo: remove accuracy block?
        # val_acc = np.mean([get_raw_accuracy(model(x.cuda()), y.cuda().long())
        #                 for x, y in test_loader
        #                 ])
        # validation_accs.append((len(accuracies), val_acc))
        # print('\nVal Loss: {:.4f}, Val Accuracy: {:.3f}'.format(val, val_acc))

        loop.close()

        # break

    print("Training Complete")