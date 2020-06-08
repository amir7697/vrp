from modules.embedding.base_embedding import BaseEmbedding


class NeighborEmbeddingTypeOne(BaseEmbedding):
    def __init__(self):
        super(NeighborEmbeddingTypeOne, self).__init__()

    def embed(self, neighbor_node, pre_node, dm, pre_capacity):
        depot = dm.get_node(0)

        if pre_capacity >= neighbor_node.demand:
            new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                             pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / pre_capacity,
                             dm.get_dis(pre_node, neighbor_node)]
        else:
            new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                             pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                             dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]

        return new_embedding
