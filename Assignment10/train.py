from custom_resnet import *
from transforms import *
from tqdm import tqdm
device ="cuda"
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Get predictions and count correct ones
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update progress bar description
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100.*correct/processed:.2f}%')

    # Print training accuracy for the epoch
    print('\nTrain set: Epoch: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Print test accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    

import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim

# Assuming the definition of your model and the Net class is provided elsewhere
model = Net().to(device)
# initial_lr = 0.001
# peak_lr = 0.01
# final_lr = 0.001

initial_lr = 0.0001
peak_lr = 0.001
final_lr = 0.0001

optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# Function to adjust learning rate
def adjust_learning_rate(optimizer, epoch, initial_lr, peak_lr, final_lr, increase_epochs, decrease_epochs):
    if epoch <= increase_epochs:
        # Linearly increase the learning rate
        lr = initial_lr + (peak_lr - initial_lr) * epoch / increase_epochs
    else:
        # Linearly decrease the learning rate
        lr = peak_lr - (peak_lr - final_lr) * (epoch - increase_epochs) / decrease_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Total epochs for increasing and decreasing learning rate
increase_epochs = 5
decrease_epochs = 19  # Total epochs after increase - increase_epochs

for epoch in range(1, 25):  # Total epochs = increase_epochs + decrease_epochs
    adjust_learning_rate(optimizer, epoch, initial_lr, peak_lr, final_lr, increase_epochs, decrease_epochs)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
