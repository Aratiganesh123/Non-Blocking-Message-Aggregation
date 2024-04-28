import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
tqdm.monitor_interval = 0

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
import random
import tqdm
import argparse

from dgi import InferenceHelper
import dgl.backend as backend

from dgl.data import RedditDataset
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler
from line_profiler import profile
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm 

class ThreeLayerGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features, allow_zero_in_degree=True)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.out_features = out_features

    def forward(self, g, features): 
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(g, x) 
        return x


class TwoLayerGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.out_features = out_features

    def forward(self, g, features):
        # Apply the first convolution with ReLU activation
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x


def load_demo_data():
    """
    Loads the OGBN-Products dataset, initializes and applies train/val/test masks to the graph.

    Returns:
        g (dgl.DGLGraph): The graph with features and masks for training, validation, and testing.
        num_classes (int): The number of classes in the dataset.
    """
  
    # data = dgl.data.CoraGraphDataset()
    # g = data[0]    
    # data = dgl.data.RedditDataset()
    # g = data[0]  # Get the graph object

    data = DglNodePropPredDataset(name='ogbn-products')
    g, labels = data[0]
    g.ndata['label'] = labels

    n_nodes = g.number_of_nodes()
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)
    n_test = n_nodes - n_train - n_val  

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    return g, data.num_classes


def node_wise_inference(model, graph, features):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions



@profile
def layer_wise_inference(model, g, features, test_nid, labels):
    """
    Perform inference using a layer-wise propagation approach, suitable for large graphs.
    
    Args:
        model: The trained GNN model.
        g: The graph on which inference is performed.
        features: Node features for the graph.
        test_nid: Node IDs for which to perform the inference.
        labels: True labels for the test nodes for accuracy computation.
    
    Prints:
        Inference time and accuracy for the test nodes.
    """
    device = features.device
    helper = InferenceHelper(model, (), batch_size=20000, device=device)
    start_time = time.time()
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

def compute_accuracy_in_batches(test_logits, test_labels, device, batch_size=1024):
    """Compute the accuracy in batches to handle large arrays efficiently."""
    num_test_nodes = test_labels.size(0)
    num_batches = (num_test_nodes + batch_size - 1) // batch_size
    total_correct = 0

    print('Num Batches', num_batches)
    print('num test nodes', num_test_nodes)
    print('test logits shape', test_logits.shape)
    print('test labels shape', test_labels.shape)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_nodes)
        batch_logits = test_logits[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]

        batch_probs = F.softmax(batch_logits, dim=1)
        batch_predictions = torch.argmax(batch_probs, dim=1)
        batch_correct = (batch_predictions == batch_labels.squeeze(1)).sum().item()
        total_correct += batch_correct


        #print(f"Batch {i}: Start {start_idx}, End {end_idx}, Correct {batch_correct}, Accumulated Correct {total_correct}")

    return total_correct






# @profile
# def adapted_inference(model, g, features, test_nid, labels, device, batch_size):
#     """
#     Perform inference on a graph node-wise, i.e., processing each node individually.
    
#     Args:
#         model: The trained GNN model.
#         graph: The graph on which inference is performed.
#         features: Node features for the graph.
    
#     Returns:
#         predictions: Predicted classes for each node in the graph.
#     """

#     start_time = time.time()
#     model.eval()  
#     model.to(device) 
    
#     g = g.to(device) 
#     features = features.to(device)  
#     labels = labels.to(device) 
#     test_nid = test_nid.to(device)  

#     out_features = model.conv2.out_features if hasattr(model.conv2, 'out_features') else model.out_features
   
#     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#     dataloader = dgl.dataloading.DataLoader(
#         g,
#         torch.arange(g.number_of_nodes()).to(device), 
#         sampler,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         device=device  
#     )

   
#     layers = [model.conv1, model.conv2, model.conv3]  

#     # Perform the computation layer by layer
#     for l, layer in enumerate(layers):
#         y = torch.zeros((g.num_nodes(), layer._out_feats if l != len(layers) - 1 else out_features), device=device)

#         # Process each batch
#         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
#             block = blocks[0]
#             input_features = features[input_nodes]
#             h = layer(block, input_features)

#             if l != len(layers) - 1:
#                 h = F.relu(h)

#             y[output_nodes] = h.detach()

#         features = y  # Output of this layer is the input to the next

#     print("Feature propagation completed. Calculating accuracy...")

#     test_logits = features[test_nid]
#     test_labels = labels[test_nid]
#     correct_predictions = compute_accuracy_in_batches(test_logits, test_labels, device, batch_size)
#     test_accuracy = correct_predictions / test_nid.size(0)
#     node_inference_time = time.time() - start_time

#     print(f"Node-wise inference time: {node_inference_time:.3f} seconds")
#     print(f"Test Accuracy: {test_accuracy:.4f} ({correct_predictions}/{test_nid.size(0)})")

#     return test_logits  # Return logits or predictions as needed


@profile
def adapted_inference(model, g, features, test_nid, labels, device, batch_size):
    """
    Perform inference on a graph node-wise, i.e., processing each node individually.
    
    Args:
        model: The trained GNN model.
        graph: The graph on which inference is performed.
        features: Node features for the graph.
    
    Returns:
        predictions: Predicted classes for each node in the graph.
    """

    start_time = time.time()
    model.eval()  
    model.to(device) 
    
    g = g.to(device) 
    features = features.to(device)  
    labels = labels.to(device) 
    test_nid = test_nid.to(device)  

    out_features = model.conv2.out_features if hasattr(model.conv2, 'out_features') else model.out_features
   
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.number_of_nodes()).to(device), 
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device  
    )

   
    layers = [model.conv1, model.conv2]  

    # Perform the computation layer by layer
    for l, layer in enumerate(layers):
        y = torch.zeros((g.num_nodes(), layer._out_feats if l != len(layers) - 1 else out_features), device=device)

        # Process each batch
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            block = blocks[0]
            input_features = features[input_nodes]
            h = layer(block, input_features)

            if l != len(layers) - 1:
                h = F.relu(h)

            y[output_nodes] = h.detach()

        features = y  # Output of this layer is the input to the next

    print("Feature propagation completed. Calculating accuracy...")

    test_logits = features[test_nid]
    test_labels = labels[test_nid]
    correct_predictions = compute_accuracy_in_batches(test_logits, test_labels, device, batch_size)
    test_accuracy = correct_predictions / test_nid.size(0)
    node_inference_time = time.time() - start_time

    print(f"Node-wise inference time: {node_inference_time:.3f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f} ({correct_predictions}/{test_nid.size(0)})")

    return test_logits  


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g, num_classes = load_demo_data()
    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].squeeze().to(device)
    
    train_mask = g.ndata['train_mask'].to(device)
    val_mask = g.ndata['val_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)
    
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0].to(device)
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0].to(device)
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0].to(device)

    # model = ThreeLayerGNN(in_features=features.shape[1], hidden_features=32, out_features=num_classes).to(device)
    model = TwoLayerGNN(in_features=features.shape[1], hidden_features=32, out_features=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    test_features = features[test_nid]

    for epoch in range(100):
        model.train()
        logits = model(g, features)
        loss = loss_function(logits[train_nid], labels[train_nid])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    model.eval()
    with torch.no_grad():
        full_logits = model(g, features).to(device)  # Ensure logits are on the same device
        val_logits = full_logits[val_nid]
        val_probs = F.softmax(val_logits, dim=1)
        val_predictions = torch.argmax(val_probs, dim=1)
        accuracy = (val_predictions == labels[val_nid]).float().mean()
        print(f'Validation Accuracy: {accuracy.item()}')


        print('Layer Wise Inference on Test Data')
        layer_wise_inference(model, g, features, test_nid, labels)

        print('--------------------------------------------------------------------')
        print('Node Wise Inference on Test Data')
        batch_size = 32
        test_predictions = adapted_inference(model, g, g.ndata['feat'], test_nid, g.ndata['label'], device, batch_size)
        node_wise_inference(model, g , features)

    



if __name__ == '__main__':
    main()
