import yaml

with open('configs/fastdvdnet_sparse.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    print(cfg)