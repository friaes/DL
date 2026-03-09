from tqdm.notebook import tqdm

import torch
import torch.nn as nn


def train(dataloader, model, n_epochs, optimizer, loss_fn, device=torch.device('cpu')):
    model.train()
    model.to(device)
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        pbar.set_description('Epoch {}, loss: {}'.format(epoch+1, loss.item()))
        

def train_mse(dataloader, model, n_epochs, device):
    train(dataloader, model, n_epochs, optimizer=torch.optim.SGD(model.parameters(), 0.001), loss_fn=nn.MSELoss(), device=device)
