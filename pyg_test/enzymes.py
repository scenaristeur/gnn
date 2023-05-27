# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print("dataset ", dataset)
#>>> ENZYMES(600)

print("len(dataset)", len(dataset))
#>>> 600

print("dataset.num_classes",dataset.num_classes)
#>>> 6

print("dataset.num_node_features",dataset.num_node_features)
#>>> 3

data = dataset[0]

print("data", data)

print("data.is_undirected()", data.is_undirected())

# shuffle and split for train and test 
dataset = dataset.shuffle()
train_dataset = dataset[:540]
test_dataset = dataset[540:]

print( "train_dataset", train_dataset, "test_dataset", test_dataset)