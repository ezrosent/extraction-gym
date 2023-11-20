//! E-graph extraction using Monte Carlo Tree Search.

use ordered_float::NotNan;

#[derive(Default)]
struct NodeHeap {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(u32);

impl NodeId {
    fn index(self) -> usize {
        self.0 as usize
    }
    fn new(index: usize) -> Self {
        Self(u32::try_from(index).unwrap())
    }
}

impl NodeHeap {
    fn get_node(&mut self, node: Node) -> NodeId {
        let res = NodeId::new(self.nodes.len());
        self.nodes.push(node);
        res
    }
}

struct State;

enum Node {
    PlaceHodlder,
    Expanded {
        state: State,
        parent: Option<NodeId>,
        children: Vec<NodeId>,
        total_cost: NotNan<f64>,
        visits: usize,
    },
}
