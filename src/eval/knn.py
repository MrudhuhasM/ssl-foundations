import torch
from torch.nn import functional as F
from eval.utils import extract_features

@torch.inference_mode()
def knn_evaluate(model, train_dataset, test_dataset, k=200, batch_size=64, temperature=0.07):

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    k = min(k, train_features.size(0))

    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    num_classes = len(train_dataset.classes)

    similarity = test_features @ train_features.T

    similarities, indices = similarity.topk(k=k, dim=1)
    train_labels_neighbors = train_labels[indices]

    weights = F.softmax(similarities / temperature, dim=1)
    one_hot_labels = F.one_hot(train_labels_neighbors, num_classes=num_classes).float()
    weighted_votes = weights.unsqueeze(2) * one_hot_labels
    class_scores = weighted_votes.sum(dim=1)
    predicted_labels = class_scores.argmax(dim=1)
    accuracy = (predicted_labels == test_labels).float().mean().item()
    return accuracy

    