import dgl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import torch
from dgi import InferenceHelper
from ogb.nodeproppred import DglNodePropPredDataset
from line_profiler import profile

class SimplifiedGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # Increase hidden dimensions or add additional parameters as needed
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features * 2, allow_zero_in_degree=True)
        self.conv3 = dgl.nn.GraphConv(hidden_features * 2, out_features, allow_zero_in_degree=True)  # Additional layer

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(g, x)  # No activation after the last layer, assume next step might be a softmax
        return x


def load_data():
    dataset = DglNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]
    g.ndata['label'] = labels
    return g, split_idx


@profile
def layer_wise_inference(model, g, features, test_nid, labels):
    device = features.device
    start_time = time.time()
    helper = InferenceHelper(model, (), batch_size=8, device=device)
    full_logits = helper.inference(g, features).to(device)
    test_logits = full_logits[test_nid]
    test_probs = F.softmax(test_logits, dim=1)
    test_predictions = torch.argmax(test_probs, dim=1)
    correct_predictions = (test_predictions == labels[test_nid]).sum().item()
    total_test_nodes = test_nid.size(0)
    test_accuracy = (test_predictions == labels[test_nid]).float().mean()
    inference_time = time.time() - start_time
    print(f"Layer Wise Inference Time: {inference_time:.3f} seconds")
    print(f"Test Accuracy: {test_accuracy.item()} ({correct_predictions}/{total_test_nodes})")

@profile
def node_wise_inference(model, g, features, test_nid, labels):
    start_time = time.time()
    predictions = model(g, features)
    probs = F.softmax(predictions, dim=1)
    predictions = torch.argmax(probs, dim=1)
    all_predictions_tensor = predictions.to(test_nid.device)
    node_predictions = all_predictions_tensor[test_nid]
    correct_predictions_node = (node_predictions == labels[test_nid]).sum().item()
    total_test_nodes_node = test_nid.size(0)
    test_accuracy_node = (node_predictions == labels[test_nid]).float().mean()
    node_inference_time = time.time() - start_time
    print(f"Node-wise inference time: {node_inference_time:.3f} seconds")
    print(f"Test Accuracy: {test_accuracy_node.item()} ({correct_predictions_node}/{total_test_nodes_node})")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g, split_idx = load_data()
    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].squeeze().to(device)
    test_nid = split_idx['test'].to(device)
    train_nid = split_idx['train'].to(device)

  
    hidden_features = 64  
    model = SimplifiedGCN(in_features=features.shape[1], hidden_features=hidden_features, out_features=g.ndata['label'].max().item() + 1).to(device)
    
    layer_wise_inference(model, g, features, test_nid, labels)
    node_wise_inference(model, g, features, test_nid, labels)

if __name__ == '__main__':
    main()