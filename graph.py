import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

@dataclass
class Vertex:
    vid: int
    people: int = 0
    kits: int = 0

@dataclass
class Edge:
    u: int
    v: int
    weight: int = 1 
    flooded: bool = False

class Graph:
    def __init__(self):
        self.vertices: Dict[int, Vertex] = {}
        self.adj: Dict[int, List[Tuple[int, Edge]]] = {}

    def add_vertex(self, vid: int, people: int = 0, kits: int = 0):
        if vid not in self.vertices:
            self.vertices[vid] = Vertex(vid, people, kits)
            self.adj[vid] = []
        else:
            self.vertices[vid].people += people
            self.vertices[vid].kits += kits

    def add_edge(self, u: int, v: int, weight: int, flooded: bool = False):
        #Force edge weight to 1 as per assignment spec
        e = Edge(u, v, 1, flooded)
        self.adj.setdefault(u, []).append((v, e))
        self.adj.setdefault(v, []).append((u, e))

    def get_edge(self, u: int, v: int) -> Optional[Edge]:
        for neighbor, edge in self.adj.get(u, []):
            if neighbor == v: return edge
        return None

    def dijkstra_dist(self, start: int, goal: int, has_kit: bool, P: int) -> float:
        """ Calculates shortest path distance considering time penalty P and flooded roads """
        pq = [(0, start)]
        distances = {start: 0}
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal: return d
            if d > distances.get(u, float('inf')): continue
            
            for v, edge in self.adj.get(u, []):
                if edge.flooded and not has_kit: continue
                # Traverse cost: P if equipped, else 1 (weight)
                cost = P if has_kit else 1
                new_dist = d + cost
                if new_dist < distances.get(v, float('inf')):
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        return float('inf')