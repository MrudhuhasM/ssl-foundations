
import torch


def linear_probe(model, train_dataset, test_dataset, num_epochs=10, batch_size=64, learning_rate=0.001):
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Add a linear classifier on top of the backbone
    sample_images, _ = next(iter(train_loader))
    sample_images = sample_images.to(device)
    sample_features = model(sample_images)
    num_features = sample_features.shape[1]
    classifier = torch.nn.Linear(num_features, len(train_dataset.classes))
    classifier.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), momentum=0.9, lr=learning_rate)

    best_accuracy = 0.0
    # Training loop
    for epoch in range(num_epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation on test set
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = model(images)
                outputs = classifier(features)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        best_accuracy = max(best_accuracy, accuracy)

    return best_accuracy