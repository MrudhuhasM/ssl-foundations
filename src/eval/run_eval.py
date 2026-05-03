from eval.models.backbone import get_encoder
from eval.data.datasets import get_dataset
from eval.linear_probe import linear_probe
from eval.knn import knn_evaluate

def main():
    # Load the encoder model
    encoder = get_encoder(checkpoint_path=None)  

    # Load datasets
    train_dataset, test_dataset = get_dataset('cifar10')

    # Evaluate using linear probe
    linear_probe_accuracy = linear_probe(encoder, train_dataset, test_dataset)
    print(f"Linear Probe Accuracy: {linear_probe_accuracy:.4f}")

    # Evaluate using k-NN
    knn_accuracy = knn_evaluate(encoder, train_dataset, test_dataset)
    print(f"k-NN Accuracy: {knn_accuracy:.4f}")