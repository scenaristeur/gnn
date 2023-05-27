# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
# need pip install pandas
from torch_geometric.datasets import OGB_MAG

dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
data = dataset[0]
print("data", data)
