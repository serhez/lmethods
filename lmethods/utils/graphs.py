from dataclasses import dataclass


@dataclass(kw_only=True)
class Node:
    id: str
    """The identifier of the node."""

    content: str
    """The content of the node."""


@dataclass(kw_only=True)
class Edge:
    id: str
    """The identifier of the edge."""

    nodes: tuple[str, str]
    """The nodes connected by the edge, referenced by their IDs."""


class DirectedGraph:
    def __init__(self, nodes: list[Node] = [], edges: list[Edge] = []):
        """
        Initializes a new directed graph.

        ### Parameters
        ----------
        `nodes`: the nodes of the graph.
        `edges`: the edges of the graph.
        """

        self._nodes = {}
        self._edges = {}

        for node in nodes:
            self.add_node(node)

        for edge in edges:
            self.add_edge(edge)

    @property
    def nodes(self) -> list[Node]:
        """The nodes of the graph."""

        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        """The edges of the graph."""

        return list(self._edges.values())

    def add_node(self, node: Node):
        """
        Adds a new node to the graph.

        ### Parameters
        ----------
        `node`: the node to add.
        """

        self._nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """
        Adds a new edge to the graph.

        ### Parameters
        ----------
        `edge`: the edge to add.
        """

        self._edges[edge.id] = edge

    def remove_node(self, id: str):
        """
        Removes a node from the graph.
        As a side effect, any edges connected to the node are also removed.

        ### Parameters
        ----------
        `id`: the ID of the node to remove.
        """

        self._nodes.pop(id)

        for edge in list(self._edges.values()):
            if edge.nodes[0] == id or edge.nodes[1] == id:
                self._edges.pop(edge.id)

    def remove_edge(self, id: str):
        """
        Removes an edge from the graph.

        ### Parameters
        ----------
        `id`: the ID of the edge to remove.
        """

        self._edges.pop(id)

    def get_node_by_id(self, id: str) -> Node:
        """
        Retrieves a node by its ID.

        ### Parameters
        ----------
        `id`: the ID of the node to retrieve.

        ### Returns
        ----------
        The node with the given ID.
        """

        return self._nodes[id]

    def get_edge_by_id(self, id: str) -> Edge:
        """
        Retrieves an edge by its ID.

        ### Parameters
        ----------
        `id`: the ID of the edge to retrieve.

        ### Returns
        ----------
        The edge with the given ID.
        """

        return self._edges[id]

    def get_adjacent_nodes(self, id: str) -> list[Node]:
        """
        Retrieves the nodes adjacent to the node with the given ID.

        ### Parameters
        ----------
        `id`: the ID of the node whose adjacent nodes to retrieve.

        ### Returns
        ----------
        The nodes adjacent to the node with the given ID.
        """

        return [
            self.get_node(edge.nodes[1])
            for edge in self._edges.values()
            if edge.nodes[0] == id
        ]

    def get_incident_edges(self, id: str) -> list[Edge]:
        """
        Retrieves the edges adjacent to the node with the given ID (incoming or outgoing).

        ### Parameters
        ----------
        `id`: the ID of the node whose adjacent edges to retrieve.

        ### Returns
        ----------
        The edges adjacent to the node with the given ID.
        """

        return [
            edge
            for edge in self._edges.values()
            if edge.nodes[0] == id or edge.nodes[1] == id
        ]

    def get_outgoing_edges(self, id: str) -> list[Edge]:
        """
        Retrieves the edges outgoing from the node with the given ID.

        ### Parameters
        ----------
        `id`: the ID of the node whose outgoing edges to retrieve.

        ### Returns
        ----------
        The edges outgoing from the node with the given ID.
        """

        return [edge for edge in self._edges.values() if edge.nodes[0] == id]

    def get_incoming_edges(self, id: str) -> list[Edge]:
        """
        Retrieves the edges incoming to the node with the given ID.

        ### Parameters
        ----------
        `id`: the ID of the node whose incoming edges to retrieve.

        ### Returns
        ----------
        The edges incoming to the node with the given ID.
        """

        return [edge for edge in self._edges.values() if edge.nodes[1] == id]


class DAG(DirectedGraph):
    def __init__(self, nodes: list[Node] = [], edges: list[Edge] = []):
        """
        Initializes a new directed acyclic graph.

        ### Parameters
        ----------
        `nodes`: the nodes of the graph.
        `edges`: the edges of the graph.

        ### Raises
        ----------
        `ValueError`: if the graph would have cycles.
        - In this case, an empty graph is created.
        """

        super().__init__(nodes, edges)

        if self._check_cycles():
            self._nodes = {}
            self._edges = {}
            raise ValueError("The graph cannot have cycles.")

    def add_node(self, node: Node):
        """
        Adds a new node to the graph.

        ### Parameters
        ----------
        `node`: the node to add.

        ### Raises
        ----------
        `ValueError`: if the graph would have cycles after adding the node.
        - The node is not added to the graph in such case.
        """

        super().add_node(node)

        if self._check_cycles():
            self._nodes.pop(node.id)
            raise ValueError("The graph cannot have cycles.")

    def add_edge(self, edge: Edge):
        """
        Adds a new edge to the graph.

        ### Parameters
        ----------
        `edge`: the edge to add.

        ### Raises
        ----------
        `ValueError`: if the graph would have cycles after adding the edge.
        - The edge is not added to the graph in such case.
        """

        super().add_edge(edge)

        if self._check_cycles():
            self._edges.pop(edge.id)
            raise ValueError("The graph cannot have cycles.")

    def get_topological_order(self) -> list[Node]:
        """
        Retrieves a topological order of the graph.

        ### Returns
        ----------
        A topological order of the graph.
        """

        in_degrees = {node.id: 0 for node in self.nodes}

        for edge in self.edges:
            in_degrees[edge.nodes[1]] += 1

        queue = [node.id for node in self.nodes if in_degrees[node.id] == 0]
        topological_order = []

        while queue:
            node_id = queue.pop(0)
            topological_order.append(self.get_node_by_id(node_id))

            for edge in self.get_outgoing_edges(node_id):
                in_degrees[edge.nodes[1]] -= 1

                if in_degrees[edge.nodes[1]] == 0:
                    queue.append(edge.nodes[1])

        return topological_order

    def _check_cycles(self) -> bool:
        """
        Checks for cycles in the graph.

        ### Returns
        ----------
        Whether the graph has cycles.
        """

        in_degrees = {node.id: 0 for node in self.nodes}

        for edge in self.edges:
            in_degrees[edge.nodes[1]] += 1

        queue = [node.id for node in self.nodes if in_degrees[node.id] == 0]
        visited = 0

        while queue:
            node_id = queue.pop(0)
            visited += 1

            for edge in self.get_outgoing_edges(node_id):
                in_degrees[edge.nodes[1]] -= 1

                if in_degrees[edge.nodes[1]] == 0:
                    queue.append(edge.nodes[1])

        return visited != len(self.nodes)
