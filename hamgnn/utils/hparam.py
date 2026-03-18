
def get_hparam_dict(config: dict = None):
    if config.setup.GNN_Net.lower() == 'dimnet':
        hparam_dict = config.representation_nets.dimnet_params
    elif config.setup.GNN_Net.lower() == 'edge_gnn':
        hparam_dict = config.representation_nets.Edge_GNN
    elif config.setup.GNN_Net.lower() == 'schnet':
        hparam_dict = config.representation_nets.SchNet
    elif config.setup.GNN_Net.lower() == 'cgcnn':
        hparam_dict = config.representation_nets.cgcnn
    elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
        hparam_dict = config.representation_nets.cgcnn_edge
    elif config.setup.GNN_Net.lower() == 'painn':
        hparam_dict = config.representation_nets.painn
    elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
        hparam_dict = config.representation_nets.cgcnn_triplet
    elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
        hparam_dict = config.representation_nets.dimenet_triplet
    elif config.setup.GNN_Net.lower() == 'dimeham':
        hparam_dict = config.representation_nets.dimeham
    elif config.setup.GNN_Net.lower() == 'dimeorb':
        hparam_dict = config.representation_nets.dimeorb
    elif config.setup.GNN_Net.lower() == 'schnorb':
        hparam_dict = config.representation_nets.schnorb
    elif config.setup.GNN_Net.lower() == 'nequip':
        hparam_dict = config.representation_nets.nequip
    elif config.setup.GNN_Net.lower() == 'hamgnn_pre':
        hparam_dict = config.representation_nets.HamGNN_pre
    elif config.setup.GNN_Net.lower()[:6] == 'hamgnn':
        hparam_dict = config.representation_nets.HamGNN_pre
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()
    for key in hparam_dict:
        if type(hparam_dict[key]) not in [str, float, int, bool, None]:
            hparam_dict[key] = type(hparam_dict[key]).__name__.split(".")[-1]
    out = {'GNN_Name': config.setup.GNN_Net}
    out.update(dict(hparam_dict))
    return out