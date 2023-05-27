# scratch
- creation venv https://docs.python.org/fr/3/library/venv.html
` python -m venv /path/to/new/virtual/environment `
- activation
```cd pyg_test
source ./bin/activate
```
 desactiver avec `deactivate`


# install
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
`pip install torch torch_geometric `
si erreur _bz2 copier un _bz2 trouvé avec locate dans le venv
install system nécessaire `sudo apt-get install libbz2-dev`
selon https://stackoverflow.com/questions/12806122/missing-python-bz2-module/61464947#61464947

`locate _bz2.`
-> /usr/lib/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so

`sudo cp /usr/lib/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so lib/python3.10/site-packages/`


-  Optional dependencies:
`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`


 # pytorch-geometric by examples
 - https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html


 # run 
 `python example1.py`