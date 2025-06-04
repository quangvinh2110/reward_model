import json
from typing import List, Optional, Set

from ..modules.node_aligner import NodeAligner


class Node:
    def __init__(
        self,
        id: str,
        step: str,
        statements: Optional[List[str]] = None,
        parent: Optional[Set["Node"]] = None,
    ):
        self.id = id
        self.step = step
        self.statements = statements 
        self.parent = parent


    def __repr__(self) -> str:
        return f"""Node(
            id={self.id},
            step={self.step},
            statements={json.dumps(self.statements, indent=2)}\n)"""


    def __str__(self) -> str:
        return self.__repr__() 
    

    def __hash__(self) -> int:
        return hash(self.id)
    

    def __eq__(self, other: "Node") -> bool:
        return isinstance(other, Node) and self.id == other.id \
            and self.step == other.step
    


class DependencyTrackingGraph:
    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        ids: Optional[List[int]] = None,
        steps: Optional[List[str]] = None,
        statements: Optional[List[List[str]]] = None,
        parents: Optional[List[Node]] = None,
    ):
        self.nodes = {}
        self.add_nodes(nodes, ids, steps, statements, parents)

    def add_one_node(
        self,
        node: Optional[Node] = None,
        id: Optional[int] = None,
        step: Optional[str] = None,
        statements: Optional[List[str]] = None,
        parent: Optional[Node] = None,
    ):
        if node is not None:
            self.nodes[node.id] = node
        elif id and step:
            self.nodes[id] = Node(
                id=id,
                step=step,
                statements=statements,
                parent=parent,
            )
        else:
            raise ValueError("Either node or id and step must be provided")
        

    def add_nodes(
        self,
        nodes: Optional[List[Node]] = None,
        ids: Optional[List[int]] = None,
        steps: Optional[List[str]] = None,
        statements: Optional[List[List[str]]] = None,
        parents: Optional[List[Node]] = None,
    ):
        if nodes:
            for node in nodes:
                self.add_one_node(node)
        elif steps:
            if ids:
                for id, step, statements, parent in zip(
                    ids, steps, statements or [None]*len(steps), parents or [None]*len(steps)
                ):
                    self.add_one_node(id=id, step=step, statements=statements, parent=parent)
            else:
                for id, (step, statements, parent) in enumerate(zip(
                    steps, statements or [None]*len(steps), parents or [None]*len(steps)
                )):
                    self.add_one_node(id=id, step=step, statements=statements, parent=parent)
        

    def add_edges(self, node_aligner: NodeAligner, threshold: float = 0.5):
        pass
    

    




    
    
