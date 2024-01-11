import torch.nn as nn
from torch.optim import Adam
import torch
from tqdm import tqdm

def run_trainer(model, train_dataloader, test_dataloader, num_epochs, device, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_history = []

    model = model.to(device)

    for epoch in range(num_epochs):
        for i, (input_tensor, target_tensor) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            # Forward pass
            output = model(input_tensor, target_tensor)

            # Compute the loss
            loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                # store train epoch loss
                loss_history.append([epoch, loss.item()])
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
    # evaluate on test loss on test dataset
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            output = model(input_tensor, target_tensor)
            loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    print(f'Test Loss: {test_loss}')
    return loss_history
