
import torch

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images)
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
    return torch.cat(features_list), torch.cat(labels_list)


def train_linear_probe(model, train_dataset, test_dataset, num_epochs=70, batch_size=64, lr = 0.01):

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    num_classes = len(train_dataset.classes)

    classifier = torch.nn.Linear(train_features.size(1), num_classes)
    classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)

    best_accuracy = 0.0

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(test_features)
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == test_labels).float().mean().item()

            if accuracy > best_accuracy:
                best_accuracy = accuracy

    return best_accuracy


def linear_probe(model, train_dataset, test_dataset, num_epochs=70, batch_size=64, lr_sweep = (0.1,0.01,0.001)):
    best_accuracy = 0.0
    best_lr = lr_sweep[0]

    for lr in lr_sweep:
        accuracy = train_linear_probe(model, train_dataset, test_dataset, num_epochs, batch_size, lr)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr

    return best_accuracy, best_lr


    