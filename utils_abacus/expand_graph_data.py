import numpy as np
import copy

def expand_graph_data(input_path, output_path, num_copies=10):
    data = np.load(input_path, allow_pickle=True)
    original_graph = data['graph'].item()
    
    expanded_graph = {}
    for i in range(num_copies):
        expanded_graph[i] = copy.deepcopy(original_graph[0])
    
    np.savez(output_path, graph=expanded_graph)
    print(f"已将数据复制{num_copies}份并保存到: {output_path}")

if __name__ == "__main__":
    input_file = "/ssd/work/whlu/HamGNN/graph/graph_data.npz"
    output_file = "/ssd/work/whlu/HamGNN/graph/graph_data_expanded.npz"
    
    expand_graph_data(input_file, output_file, num_copies=10)
