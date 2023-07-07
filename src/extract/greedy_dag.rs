//! A variant of the greedy algorithm that greedily minimizes DAG cost, rather
//! than tree cost.
//!
//! To do this, the algorithm keeps track of the set of nodes _and_ their
//! optimal cost, rather than just the minimum cost of each node. We use a
//! variant of HAMTs with PATRICIA-style unions (with incrementally computed
//! aggregates) to make this efficient: larger benchmarks are 2-3x faster using
//! this data-structure. That data-structure is in the `val-trie` crate.
//!
//! The current implementation here is fairly simplistic, and a lot of unions
//! are recomputed unnecessarily; there is a lot of room for improvement here in
//! terms of runtime. Still, the largest examples in the data-set finish in a
//! few seconds.

use std::rc::Rc;

use indexmap::IndexSet;
use ordered_float::NotNan;

use crate::{Class, ExtractionResult, Extractor, Id, Node, SimpleEGraph};

pub(crate) struct GreedyDagExtractor;

impl Extractor for GreedyDagExtractor {
    fn extract(&self, egraph: &SimpleEGraph, _roots: &[Id]) -> ExtractionResult {
        let node_centric = NodeCentricEgraph::new(egraph);
        node_centric.extract()
    }
}

/// We make extensive use of "ENode Ids" in this algorithm. SimpleEGraph is
/// "E-class" centric, so we keep a separate, flatter, struct around.
struct NodeCentricEgraph {
    nodes: IndexSet<Node>,
    classes: Vec<Class>,
    // Indexed by NodeId
    node_costs: Rc<Vec<NotNan<f64>>>,
}

type NodeId = usize;

impl NodeCentricEgraph {
    fn new(egraph: &SimpleEGraph) -> NodeCentricEgraph {
        let mut classes = Vec::with_capacity(egraph.classes.len());
        let mut nodes = IndexSet::new();
        for (_, (_, class)) in egraph.classes.iter().enumerate() {
            classes.push(class.clone());
            for node in &class.nodes {
                nodes.insert(node.clone());
            }
        }
        let mut node_costs = Vec::with_capacity(nodes.len());
        for node in &nodes {
            node_costs.push(node.cost);
        }
        NodeCentricEgraph {
            classes,
            nodes,
            node_costs: Rc::new(node_costs),
        }
    }

    fn empty_node_set(&self) -> NodeSet {
        NodeSet {
            trie: Default::default(),
            costs: self.node_costs.clone(),
        }
    }

    fn compute_cost(&self, node: &Node, costs: &[Option<NodeSet>]) -> Option<NodeSet> {
        let node_id = self.nodes.get_index_of(node).unwrap();
        let mut init = self.empty_node_set();
        init.add(node_id);
        node.children
            .iter()
            .map(|&child| &costs[child])
            .try_fold(init, |mut acc, child| {
                let child = child.as_ref()?;
                acc.union(child);
                Some(acc)
            })
    }

    fn extract(&self) -> ExtractionResult {
        let mut result = ExtractionResult::new(self.classes.len());
        let mut costs = Vec::<Option<NodeSet>>::new();
        costs.resize_with(self.classes.len(), || None);
        let mut did_something = false;

        loop {
            for (i, class) in self.classes.iter().enumerate() {
                for (j, node) in class.nodes.iter().enumerate() {
                    let new_cost = self.compute_cost(node, &costs);
                    match (&costs[i], new_cost) {
                        (None, None) | (Some(_), None) => {}
                        (None, x @ Some(_)) => {
                            costs[i] = x;
                            result.choices[i] = j;
                            did_something = true;
                        }
                        (Some(cur), Some(new)) => {
                            if new.cost() < cur.cost() {
                                costs[i] = Some(new);
                                result.choices[i] = j;
                                did_something = true;
                            }
                        }
                    }
                }
            }

            if did_something {
                did_something = false;
            } else {
                break;
            }
        }

        result
    }
}

/// The `_agg` APIs in val-trie are fairly low-level and easy to misuse.
/// `NodeSet` is a wrapper that exposes the minimal API required for this
/// algorithm to work.
#[derive(Clone)]
struct NodeSet {
    trie: val_trie::HashSet<NodeId, AddNotNan>,
    costs: Rc<Vec<NotNan<f64>>>,
}

impl NodeSet {
    fn add(&mut self, node: NodeId) {
        self.trie
            .insert_agg(node, |node| AddNotNan(self.costs[*node]));
    }

    fn union(&mut self, other: &Self) {
        self.trie
            .union_agg(&other.trie, |node| AddNotNan(self.costs[*node]));
    }

    fn cost(&self) -> NotNan<f64> {
        self.trie.agg().0
    }
}

#[derive(Default, Copy, Clone)]
struct AddNotNan(NotNan<f64>);
impl val_trie::Group for AddNotNan {
    fn add(&mut self, other: &Self) {
        self.0 += other.0;
    }

    fn inverse(&self) -> Self {
        AddNotNan(-self.0)
    }

    fn sub(&mut self, other: &Self) {
        self.0 -= other.0;
    }
}
