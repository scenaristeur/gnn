from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print("batch",batch)
   # >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    print("batch.num_graphs",batch.num_graphs)
   # >>> 32


for data in loader:
    print("data",data)
    #>>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    print("data.num_graphs",data.num_graphs)
    #>>> 32

    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    print("x.size()",x.size())
    #>>> torch.Size([32, 21])