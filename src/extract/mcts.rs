//! E-graph extraction using Monte Carlo Tree Search.
//!
//! NOTE: there's a lot here that's very unoptimized. The goal here is to get a
//! clean implementation that uses the existing extraction-gym abstractions.
//! Here's a nonexhaustive list of things that could be improved:
//!
//! 1. NodeId and ClassId can just be integers; hashing and cloning become much
//! cheaper then compared with `Arc<str>`.
//! 2. Enumerations of available moves and completeness checks can be reused
//! across iterations with the right data-structure.
//! 3. State representation is likely not optimal.
//! 4. There is a lot of "symmetry" when picking some e-nodes, where for some
//! nodes in different e-classes (though surely not all), we can pick them in
//! either order. We can probably reuse work there too (Wikipedia talks about
//! RAVE).
#![allow(unused)]
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    rc::Rc,
};

use egraph_serialize::{ClassId, EGraph, NodeId};
use im_rc::{HashMap as ImHashMap, HashSet as ImHashSet};
use ordered_float::NotNan;

use crate::Cost;

#[derive(Default)]
struct NodeHeap {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TreeNodeId(u32);

impl TreeNodeId {
    fn index(self) -> usize {
        self.0 as usize
    }
    fn new(index: usize) -> Self {
        Self(u32::try_from(index).unwrap())
    }
}

impl NodeHeap {
    fn get_node(&mut self, node: Node) -> TreeNodeId {
        let res = TreeNodeId::new(self.nodes.len());
        self.nodes.push(node);
        res
    }
}

#[derive(Clone)]
struct State {
    remaining: ImHashSet<NodeId>,
    choices: ImHashMap<ClassId, NodeId>,
}

impl State {
    /// Get the cost of the state.
    ///
    /// A cost is only returned if the given choices define a valid extraction
    /// for the egraph.
    fn evaluate(&self, eg: &EGraph) -> Option<Cost> {
        fn evaluate_node(
            node_id: &NodeId,
            state: &State,
            eg: &EGraph,
            seen: &mut HashMap<ClassId, bool>,
        ) -> Option<Cost> {
            let node = &eg.nodes[node_id];
            let mut cost = node.cost;
            for other in &node.children {
                let class = eg.nid_to_cid(other);
                cost += evaluate_class(class, state, eg, seen)?;
            }
            Some(cost)
        }
        fn evaluate_class(
            class_id: &ClassId,
            state: &State,
            eg: &EGraph,
            seen: &mut HashMap<ClassId, bool>,
        ) -> Option<Cost> {
            match seen.entry(class_id.clone()) {
                Entry::Occupied(o) => {
                    return if *o.get() {
                        Some(Default::default())
                    } else {
                        // Cycle
                        None
                    };
                }
                Entry::Vacant(mut v) => v.insert(false),
            };
            let choice = state.choices.get(class_id)?;
            let res = evaluate_node(choice, state, eg, seen)?;
            *seen.get_mut(class_id).unwrap() = true;
            Some(res)
        }
        let mut seen = Default::default();
        let mut cost = Default::default();
        for class_id in &eg.root_eclasses {
            cost += evaluate_class(class_id, self, eg, &mut seen)?;
        }
        Some(cost)
    }

    fn pick_unvisited_node(&mut self, eg: &EGraph) -> Option<NodeId> {
        // XXX: we should be picking this node randomly, but using the first one
        // "per the hash function" _may_ be good enough to start with.
        // Unforunately we can probably get "stuck" with an unlucky hash seed.
        // We'll want to fix this.
        let node = self.remaining.iter().next()?.clone();
        self.remaining.remove(&node);
        let class = eg.nid_to_class(&node);
        for other in class.nodes.iter() {
            self.remaining.remove(other);
        }
        self.choices.insert(class.id.clone(), node.clone());
        Some(node)
    }

    fn run_simulation(&self, eg: &EGraph) -> Option<Cost> {
        let mut local_copy = self.clone();
        while local_copy.pick_unvisited_node(eg).is_some() {}
        local_copy.evaluate(eg)
    }
}

enum Node {
    PlaceHodlder(State),
    Expanded {
        state: State,
        parent: Option<TreeNodeId>,
        children: HashMap<NodeId, TreeNodeId>,
        total_cost: Cost,
        visits: usize,
    },
}

enum ExpandResult {
    AlreadyExpanded,
    Expanded,
}

impl Node {
    fn expand(&mut self, parent: Option<TreeNodeId>) -> ExpandResult {
        match self {
            Node::PlaceHodlder(state) => {
                *self = Node::Expanded {
                    state: state.clone(),
                    parent,
                    children: Default::default(),
                    total_cost: Default::default(),
                    visits: 0,
                };
                ExpandResult::Expanded
            }
            Node::Expanded { .. } => ExpandResult::AlreadyExpanded,
        }
    }
}
