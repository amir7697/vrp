from modules.embedding.base_embedding import BaseEmbedding


class NodeEmbedding(BaseEmbedding):
    def __init__(self):
        super(NodeEmbedding, self).__init__()

    def embed(self, node, capacity):
        if node.capacity == 0:
            node.embedding = [node.x, node.y, node.demand * 1.0 / capacity, node.px, node.py, 0.0, node.dis]
        else:
            node.embedding = [node.x, node.y, node.demand * 1.0 / capacity, node.px, node.py,
                              node.demand * 1.0 / node.capacity, node.dis]