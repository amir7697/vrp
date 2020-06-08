from modules.embedding.base_embedding import BaseEmbedding


class NeighborEmbeddingTypeTwo(BaseEmbedding):
    def __init__(self):
        super(NeighborEmbeddingTypeTwo, self).__init__()

    def embed(self, neighbor_node, pre_node, cur_node, dm, pre_capacity):
        depot = dm.get_node(0)

        if pre_capacity >= neighbor_node.demand:
            new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x),
                             (neighbor_node.y - depot.y) * (pre_node.y - depot.y),
                             (neighbor_node.demand - cur_node.demand) * 1.0 / pre_capacity, pre_node.px,
                             pre_node.py, (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity,
                             dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
        else:
            new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x),
                             (neighbor_node.y - depot.y) * (pre_node.y - depot.y),
                             (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity, pre_node.px,
                             pre_node.py, (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity,
                             dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]

        return new_embedding
