import torch
from torch_sparse import SparseTensor

def triplets(edge_index, num_nodes, cell_shift):
    row, col = edge_index  # j->i
            
    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
                    
    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
                   
    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()
                       
    """
    idx_i -> pos[idx_i]
    idx_j -> pos[idx_j] - nbr_shift[idx_ji]
    idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
    """
    # Remove i == k triplets with the same cell_shift.
    relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
    mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
    idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]
               
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji