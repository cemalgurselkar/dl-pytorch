import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils import *

#Activation Functions
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

class Perceptron(nn.Module):
    """
    Single-layer perceptron for binary classification
    """
    def __init__(self,input_size):
        super().__init__()
        self.fc = nn.Linear(input_size,1)
    
    def forward(self,x):
        x = self.fc(x)
        return torch.sigmoid(x)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (2 hidden layers) for classification
    """
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x


def main(args):
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    torch.manual_seed(0)
    input_data = torch.rand(1000, args.input_size)
    labels = (input_data.sum(axis=1) > args.input_size / 2).float().unsqueeze(1)  # Binary Classification
    dataset = TensorDataset(input_data, labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Select model
    if args.model == "perceptron":
        model = Perceptron(input_size=args.input_size).to(device)
        criterion = nn.BCELoss()  # Binary Cross-Entropy for perceptron
    elif args.model == "mlp":
        model = MLP(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size).to(device)
        criterion = nn.CrossEntropyLoss()  # Cross-Entropy for MLP
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train and test
    train_losses = train_model(model, train_loader, criterion, optimizer, device, args.epochs)
    test_accuracies = []
    for epoch in range(args.epochs):
        test_accuracies.append(test_model(model, test_loader, device))

    # Plot and save metrics
    plot_metrics(train_losses, test_accuracies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Perceptron or MLP on synthetic data")
    parser.add_argument("--model", type=str, default="perceptron", choices=["perceptron", "mlp"], help="Model type (default: perceptron)")
    parser.add_argument("--input_size", type=int, default=10, help="Input size (default: 10)")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size for MLP (default: 64)")
    parser.add_argument("--output_size", type=int, default=2, help="Output size for MLP (default: 2)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    args = parser.parse_args()

    main(args)