import numpy as np

from modules.embedding.node_embedding import NodeEmbedding


class VrpNode(object):
    def __init__(self, x, y, demand, px, py, capacity, dis, embedding=None):
        self.x = x
        self.y = y
        self.demand = demand
        self.px = px
        self.py = py
        self.capacity = capacity
        self.dis = dis
        if embedding is None:
            self.embedding = None
        else:
            self.embedding = embedding.copy()


class VrpManager(object):
    def __init__(self, capacity):
        self.nodes = []
        self.num_nodes = 0
        self.capacity = capacity
        self.route = []
        self.vehicle_state = []
        self.tot_dis = []
        self.node_embedding = NodeEmbedding()
        self.encoder_outputs = None

    def get_node(self, idx):
        return self.nodes[idx]

    def clone(self):
        res = VrpManager(self.capacity)
        res.nodes = []
        for i, node in enumerate(self.nodes):
            res.nodes.append(VrpNode(x=node.x, y=node.y, demand=node.demand, px=node.px, py=node.py,
                                     capacity=node.capacity, dis=node.dis, embedding=node.embedding))
        res.num_nodes = self.num_nodes
        res.route = self.route[:]
        res.vehicle_state = self.vehicle_state[:]
        res.tot_dis = self.tot_dis[:]
        res.encoder_outputs = self.encoder_outputs.clone()
        return res

    def get_dis(self, node_1, node_2):
        return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)

    def get_neighbor_idxes(self, route_idx):
        neighbor_idxes = []
        route_node_idx = self.vehicle_state[route_idx][0]
        pre_node_idx, pre_capacity = self.vehicle_state[route_idx - 1]
        for i in range(1, len(self.vehicle_state) - 1):
            cur_node_idx = self.vehicle_state[i][0]
            if route_node_idx == cur_node_idx:
                continue
            if pre_node_idx == 0 and cur_node_idx == 0:
                continue
            cur_node = self.get_node(cur_node_idx)
            if route_node_idx == 0 and i > route_idx and cur_node.demand > pre_capacity:
                continue
            neighbor_idxes.append(i)
        return neighbor_idxes

    def add_route_node(self, node_idx):
        node = self.get_node(node_idx)

        if len(self.vehicle_state) == 0:
            pre_node_idx = 0
            pre_capacity = self.capacity
        else:
            pre_node_idx, pre_capacity = self.vehicle_state[-1]

        pre_node = self.get_node(pre_node_idx)
        if node_idx > 0:
            self.vehicle_state.append((node_idx, pre_capacity - self.nodes[node_idx].demand))
        else:
            self.vehicle_state.append((node_idx, self.capacity))

        cur_dis = self.get_dis(node, pre_node)
        if len(self.tot_dis) == 0:
            self.tot_dis.append(cur_dis)
        else:
            self.tot_dis.append(self.tot_dis[-1] + cur_dis)
        new_node = VrpNode(x=node.x, y=node.y, demand=node.demand, px=pre_node.x, py=pre_node.y, capacity=pre_capacity,
                           dis=cur_dis)
        self.node_embedding.embed(new_node, self.capacity)
        self.nodes[node_idx] = new_node
        self.route.append(new_node.embedding[:])
