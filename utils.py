import matplotlib.pyplot as plt
import torch


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """
    Training loop for the model.
    """
    model.train()
    train_losses = []  # Store loss values for each epoch
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    return train_losses


def test_model(model, test_loader, device):
    """
    Testing loop for the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert outputs to predicted class
            predicted = torch.round(torch.sigmoid(outputs) if outputs.size(1) == 1 else torch.argmax(outputs, dim=1))
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_metrics(train_losses, test_accuracies, output_path="metrics.png"):
    """
    Plot and save training loss and test accuracy graphs.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Graphs saved to {output_path}")
