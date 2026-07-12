import numpy as np

class Node:
    def __init__(self, ID, x, y, demand):
        self.ID = ID
        self.x = x
        self.y = y
        self.demand = demand
        self.inroute = None
        self.isInterior = False
        self.dnEdge = None
        self.ndEdge = None

class Edge:

    def __init__(self,origin,end,cost):
        self.origin = origin
        self.end = end
        self.cost = cost
        self.savings = 0.0
        self.invEdge = None


class Route:

    def __init__(self):
        self.cost = 0.0
        self.edges = []
        self.demand = 0.0

    def reverse(self):
        size=len(self.edges)
        for i in range(size):
            edge = self.edges[i]
            invEdge = edge.invEdge
            self.edges.remove(edge)
            self.edges.insert(0,invEdge)

    def __repr__(self):
       if not self.edges:
           return f"Route(empty, cost={self.cost:.2f}, demand={self.demand:.2f})"

       path_nodes = [self.edges[0].origin.ID] + [edge.end.ID for edge in self.edges]
       path_str = " - ".join(map(str, path_nodes))
       return f"Route(path={path_str} | cost={self.cost:.2f}| demand={self.demand:.2f})"

class Solution:
    last_ID = -1

    def __init__(self):
        Solution.last_ID += 1
        self.ID = Solution.last_ID
        self.routes = []
        self.cost = 0.0
        self.demand = 0.0
