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