import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import tqdm
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.linear3 = nn.Linear(1600, 64)
        self.linear4 = nn.Linear(64, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.log_softmax(x)
        return x


def train():
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    transformT = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomAffine(
                10, translate=(0.1, 0.1), scale=(0.6, 1.2), shear=10
            ),
            v2.RandomHorizontalFlip(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    trainset = datasets.MNIST("mnist", download=True, train=True, transform=transformT)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST("mnist", download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=5)

    epochs = 15

    loadingBar = tqdm.tqdm(range(epochs), desc="Epochs")
    loadingBar.total = trainloader.__len__()

    model.train()

    losses = []

    fig, ax = plt.subplots()

    (line,) = ax.plot(losses)
    (lrLine,) = ax.plot([0, len(losses)], [0.01, 0.01], "r--")

    plt.ion()
    plt.show()

    for epoch in range(epochs):
        loadingBar.desc = f"Epoch {epoch + 1}"
        loadingBar.reset()
        runningLoss = 0
        for i, (images, labels) in enumerate(trainloader):
            loadingBar.update(1)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()

            runningLoss += loss.item() * images.size(0)

            optimizer.step()
            losses.append(runningLoss / ((i + 1) * images.size(0)))

            # Update the line object and redraw the figure every 100 iterations
            if i % 100 == 0:
                line.set_ydata(losses)
                line.set_xdata(range(len(losses)))

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        if loss.item() < 0.01:
            break

        lrLine.set_ydata(
            [
                optimizer.param_groups[0]["lr"] * 1000,
                optimizer.param_groups[0]["lr"] * 1000,
            ]
        )
        lrLine.set_xdata([0, len(losses)])

        #scheduler.step()

    model.eval()

    incorrect = []
    correctL = []

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if predicted != labels:
                incorrect.append((images, labels, predicted))
            else:
                correctL.append((images, labels, predicted))

    print(f"Accuracy: {correct / total}")

    torch.save(model.state_dict(), "mnistConv01.pth")


if __name__ == "__main__":
    train()
