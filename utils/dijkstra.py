# utils/dijkstra.py

import networkx as nx

def dijkstra_route(graph, src, dst):
    try:
        path = nx.dijkstra_path(graph, src, dst)
        return path
    except nx.NetworkXNoPath:
        return []