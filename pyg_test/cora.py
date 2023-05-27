# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
#>>> Cora()
print("Cora dataset",dataset)

print("len(dataset)",len(dataset))
#>>> 1

print("dataset.num_classes",dataset.num_classes)
#>>> 7

print("dataset.num_node_features",dataset.num_node_features)
#>>> 1433

data = dataset[0]
print("data",data)
#>>> Data(edge_index=[2, 10556], test_mask=[2708],
#        train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

print("data.is_undirected()",data.is_undirected())
#>>> True

print("data.train_mask.sum().item()",data.train_mask.sum().item())
#>>> 140

print("data.val_mask.sum().item()",data.val_mask.sum().item())
#>>> 500

print("data.test_mask.sum().item()",data.test_mask.sum().item())
#>>> 1000

for key, item in data:
    print(f'key {key}, item {item} found in data')