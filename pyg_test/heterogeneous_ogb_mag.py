# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
# need pip install pandas
from torch_geometric.datasets import OGB_MAG

dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
data = dataset[0]
print("data", data)

paper_node_data = data['paper']
cites_edge_data = data['paper', 'cites', 'paper']
#cites_edge_data = data['paper', 'paper']
#cites_edge_data = data['cites']



#print("paper_node_data",paper_node_data)

print("data['paper'].year", data['paper'].year)
print("data['paper'].x", data['paper'].x)
#print("data['paper'].x['author']", data['paper'].x['author'])

node_types, edge_types = data.metadata()
print(node_types)
#['paper', 'author', 'institution']
print(edge_types)
#[('paper', 'cites', 'paper'),
#('author', 'writes', 'paper'),
#('author', 'affiliated_with', 'institution')]

homogeneous_data = data.to_homogeneous()
print(homogeneous_data)