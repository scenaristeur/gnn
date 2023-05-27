import torch
from torch_geometric.data import Data

# option 1
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index)
# #>>> Data(edge_index=[2, 4], x=[3, 1])

# print(data)


# Note that edge_index, i.e. the tensor defining the source and target nodes of all edges, is not a list of index tuples. If you want to write your indices this way, you should transpose and call contiguous on it before passing them to the data constructor:

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

data.validate(raise_on_error=True)
print("Data validated",data)

# validation

# edge_indexE = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 8], # 8 au lieu de 2
#                            [2, 1]], dtype=torch.long)
# xE = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# dataE = Data(x=xE, edge_index=edge_indexE.t().contiguous())


# print(dataE, "should give an error")
# dataE.validate(raise_on_error=True)


# utility functions 

print("data.keys: ",data.keys)
#>>> ['x', 'edge_index']

print("data['x']",data['x'])
#>>> tensor([[-1.0],
#            [0.0],
#            [1.0]])

for key, item in data:
    print(f'{key}, {item} found in data')
#>>> x found in data
#>>> edge_index found in data

print("'edge_attr' in data",'edge_attr' in data)
#>>> False

print("data.num_nodes",data.num_nodes)
#>>> 3

print("data.num_edges",data.num_edges)
#>>> 4

print("data.num_node_features",data.num_node_features)
#>>> 1

print("data.has_isolated_nodes()",data.has_isolated_nodes())
#>>> False

print("data.has_self_loops()",data.has_self_loops())
#>>> False

print("data.is_directed()",data.is_directed())
#>>> False

# Transfer data object to GPU.
print("cuda available? ",torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(" transfert data object to GPU device",device)
    data = data.to(device)
else:
    print("no cuda available, using CPU")

print("READY",data)