# Import the shit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms

from model import TransformerNet

import numpy as np
# import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Data loading 
    # sort sentences by length (todo: do this in prepare_data.py when you write the sentences to file)
    # apply transforms as necessary (pad here?)
    en_data = sorted(en_data, key=lambda x: len(x))
    jp_data = sorted(jp_data, key=lambda x: len(x))

    # todo: redo all of this below this line
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

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
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()  # Backpropagation
            optimizer.step()  # Apply the gradients

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training Complete")