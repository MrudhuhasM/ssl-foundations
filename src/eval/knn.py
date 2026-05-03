import torch
from torch.nn import functional as F

@torch.inference_mode()
def knn_evaluate(model, train_dataset, test_dataset, k=200, batch_size=64):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract features for training data
    train_features, train_labels = [], []

    for images, labels in train_loader:
        images = images.to(device)
        features = model(images)
        train_features.append(features)
        train_labels.append(labels)
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)

    # Extract features for test data
    test_features, test_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        features = model(images)
        test_features.append(features)
        test_labels.append(labels)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    similarity = test_features @ train_features.T
    
    topk = torch.topk(similarity, k=k, dim=1)
    topk_indices = topk.indices
    topk_labels = train_labels[topk_indices]
    topk_similarities = topk.values

    temperature = 0.1
    weights = F.softmax(topk_similarities / temperature, dim=1)
    num_classes = len(train_dataset.classes)
    class_scores = torch.zeros(test_features.size(0), num_classes, device=device)
    for i in range(k):
        class_scores.scatter_add_(1, topk_labels[:, i:i+1], weights[:, i:i+1])
    predicted_labels = torch.argmax(class_scores, dim=1)
    accuracy = (predicted_labels == test_labels.to(device)).float().mean().item()
    return accuracy
    

  