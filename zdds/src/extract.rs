//! Routines for extracting DAGs of ENodes from an EGraph.
use std::{cmp, hash::Hash, mem, time::Instant};

use hashbrown::hash_map::Entry;
use log::debug;
use ordered_float::NotNan;
use petgraph::{prelude::NodeIndex, stable_graph::StableDiGraph, visit::Dfs};

use crate::{
    choose_nodes,
    egraph::{Cost, Pool},
    Egraph, HashMap, HashSet,
};

/// The type used to return DAGs of expressions during extraction.
///
/// This is just a type alias for the underlying petgraph type, which is a
/// general graph rather than an acylic one. We use the "Stable" variant because
/// we remove any unreachable nodes from the DAG before returning it.
pub type Dag<T> = StableDiGraph<T, usize>;

/// The output "term" returned by an extaction procedure, represented as a
/// graph.
pub struct ExtractResult<T> {
    pub root: NodeIndex,
    pub dag: Dag<T>,
    pub total_cost: Cost,
}

pub fn extract_greedy<E: Egraph>(
    egraph: &mut E,
    root: E::EClassId,
) -> Option<ExtractResult<E::ENodeId>> {
    extract(egraph, root, NullFilter)
}

pub fn extract_zdd<E: Egraph>(
    egraph: &mut E,
    root: E::EClassId,
    node_limit: Option<usize>,
) -> Option<ExtractResult<E::ENodeId>> {
    debug!("attempting zdd extraction");
    let (nodes, _) = choose_nodes(egraph, root.clone(), node_limit)?;
    debug!("zdd narrowed extractions to {} nodes", nodes.len());
    let extract_result = extract(egraph, root, SetFilter(nodes.iter().cloned().collect()))?;
    Some(extract_result)
}

pub(crate) fn extractor<'a, E: Egraph, F: ENodeFilter<E::ENodeId>>(
    egraph: &'a mut E,
    root: &E::EClassId,
    filter: F,
    pool: &Pool<E>,
) -> Extractor<'a, E, F> {
    let mut extractor = Extractor {
        graph: Default::default(),
        nodes: Default::default(),
        best: Default::default(),
        cost_guess: Default::default(),
        egraph,
        filter,
    };
    let start = Instant::now();
    extractor.traverse_class(root.clone(), pool);
    let initial_traversal_done = Instant::now();
    extractor.iterate(pool);
    let iteration_done = Instant::now();
    log::debug!(
        "root node entry = {:?}",
        extractor.best.get(root).map(|(_, cost)| cost),
    );
    log::debug!(
        "traversal took={:?}",
        initial_traversal_done.checked_duration_since(start),
    );
    log::debug!(
        "iteration took={:?}",
        iteration_done.checked_duration_since(initial_traversal_done)
    );
    extractor
}

// TODO: re-do greedy, make it simpler to start with. Then make it
// cost-parametric from the start.
// _then_ see if we can do the "top-down seeding" or if that's too much

pub(crate) fn greedy_costs<E: Egraph>(
    egraph: &mut E,
    root: &E::EClassId,
) -> HashMap<E::EClassId, Cost> {
    let extractor = extractor(egraph, root, NullFilter, &Pool::default());
    extractor
        .best
        .into_iter()
        .map(|(k, (_, v))| (k, v))
        .collect()
}

pub(crate) fn extract<E: Egraph, F: ENodeFilter<E::ENodeId>>(
    egraph: &mut E,
    root: E::EClassId,
    filter: F,
) -> Option<ExtractResult<E::ENodeId>> {
    let pool = Pool::default();
    let extractor = extractor(egraph, &root, filter, &pool);
    extractor.into_result(&root, &pool)
}

pub(crate) struct Extractor<'a, E: Egraph, Filter> {
    graph: Dag<E::ENodeId>,
    nodes: HashMap<E::ENodeId, (NodeIndex, Cost)>,
    best: HashMap<E::EClassId, (E::ENodeId, Cost)>,
    cost_guess: HashMap<E::ENodeId, Option<Cost>>,
    egraph: &'a mut E,
    filter: Filter,
}

impl<'a, E: Egraph, Filter: ENodeFilter<E::ENodeId>> Extractor<'a, E, Filter> {
    fn into_result(
        mut self,
        root: &E::EClassId,
        pool: &Pool<E>,
    ) -> Option<ExtractResult<E::ENodeId>> {
        let root_node = self.build_graph(root, pool, &mut Default::default())?;
        let total_cost = self.prune_and_compute_cost(root_node);
        Some(ExtractResult {
            root: root_node,
            dag: self.graph,
            total_cost,
        })
    }

    fn prune_and_compute_cost(&mut self, root: NodeIndex) -> Cost {
        // We don't want to use the cost in `hashcons` because it can
        // double-count nodes that have multiple parents in the DAG.
        let mut cost = Cost::default();
        let mut visited = HashSet::default();
        let mut dfs = Dfs::new(&self.graph, root);
        while let Some(n) = dfs.next(&self.graph) {
            visited.insert(n);
            cost += self.egraph.cost(self.graph.node_weight(n).unwrap());
        }
        self.graph.retain_nodes(|_, x| visited.contains(&x));
        log::debug!("final cost={cost}");
        cost
    }

    fn iterate(&mut self, pool: &Pool<E>) {
        let mut changed = true;
        let cost_guess = mem::take(&mut self.cost_guess);
        let mut classes = pool.class_vec();
        self.egraph
            .for_each_class(|class| classes.push(class.clone()));
        while changed {
            changed = false;
            for class in classes.iter() {
                changed |= self.update_class(class, pool);
            }
        }
        self.cost_guess = cost_guess;
    }

    fn build_graph(
        &mut self,
        root: &E::EClassId,
        pool: &Pool<E>,
        visited: &mut HashMap<E::EClassId, NodeIndex>,
    ) -> Option<NodeIndex> {
        if let Some(node) = visited.get(root) {
            return Some(*node);
        }
        let root_node = self.best.get(root)?.0.clone();
        let res = self
            .nodes
            .get(&root_node)
            .expect("nodes must exist when they represent a class")
            .0;
        let mut classes = pool.class_vec();
        self.egraph.get_children(&root_node, &mut classes);
        for (i, class) in classes.drain(..).enumerate() {
            let neighbor = self
                .build_graph(&class, pool, visited)
                .expect("child classes must exist if node does");
            self.graph.add_edge(res, neighbor, i);
        }
        visited.insert(root.clone(), res);
        Some(res)
    }

    fn traverse_class(&mut self, class: E::EClassId, pool: &Pool<E>) -> Option<Cost> {
        let mut nodes = pool.node_vec();
        self.egraph.expand_class(&class, &mut nodes);
        self.filter.filter(&mut nodes);
        let (node, cost) = nodes
            .drain(..)
            .filter_map(|node| {
                let cost = self.traverse_node(node.clone(), pool)?;
                Some((node, cost))
            })
            .min_by_key(|(_, cost)| *cost)?;
        match self.best.entry(class) {
            Entry::Occupied(mut o) => {
                if o.get().1 > cost {
                    *o.get_mut() = (node, cost);
                }
            }
            Entry::Vacant(v) => {
                v.insert((node, cost));
            }
        }
        Some(cost)
    }

    fn traverse_node(&mut self, node: E::ENodeId, pool: &Pool<E>) -> Option<Cost> {
        if let Some(x) = self.cost_guess.get(&node) {
            return *x;
        };
        self.cost_guess.insert(node.clone(), None);
        let mut classes = pool.class_vec();
        self.egraph.get_children(&node, &mut classes);
        let mut total_cost = self.egraph.cost(&node);
        for class in classes.drain(..) {
            let cost = self.traverse_class(class, pool)?;
            total_cost += cost
        }
        let graph_node = self.graph.add_node(node.clone());
        let res = Some(total_cost);
        self.nodes.insert(node.clone(), (graph_node, total_cost));
        self.cost_guess.insert(node, res);
        res
    }

    fn update_class(&mut self, class: &E::EClassId, pool: &Pool<E>) -> bool {
        let mut nodes = pool.node_vec();
        self.egraph.expand_class(class, &mut nodes);
        self.filter.filter(&mut nodes);
        let Some((node, (_, cost))) = nodes
            .drain(..)
            .filter_map(|node| {
                let mapping = self.update_node(&node, pool)?;
                Some((node, mapping))
            })
            .min_by_key(|(_, (_, cost))| *cost)
        else {
            return false;
        };
        match self.best.entry(class.clone()) {
            Entry::Occupied(mut o) => {
                let (cur_node, cur_cost) = o.get_mut();
                if *cur_cost > cost {
                    *cur_node = node;
                    *cur_cost = cost;
                    true
                } else {
                    false
                }
            }
            Entry::Vacant(v) => {
                v.insert((node, cost));
                true
            }
        }
    }

    fn update_node(&mut self, node: &E::ENodeId, pool: &Pool<E>) -> Option<(NodeIndex, Cost)> {
        let mut classes = pool.class_vec();
        self.egraph.get_children(node, &mut classes);
        let mut cost = self.egraph.cost(node);
        for class in classes.iter() {
            let (_, child_cost) = self.best.get(class)?;
            cost += child_cost;
        }
        match self.nodes.entry(node.clone()) {
            Entry::Occupied(mut o) => {
                let (node, cur_cost) = o.get_mut();
                if *cur_cost > cost {
                    *cur_cost = cost;
                }
                Some((*node, *cur_cost))
            }
            Entry::Vacant(v) => {
                let graph_node = self.graph.add_node(node.clone());
                let res = (graph_node, cost);
                v.insert(res);
                Some(res)
            }
        }
    }
}

pub(crate) trait ENodeFilter<T> {
    fn filter(&self, enodes: &mut Vec<T>);
}

pub(crate) struct NullFilter;

impl<T> ENodeFilter<T> for NullFilter {
    fn filter(&self, _: &mut Vec<T>) {}
}

pub(crate) struct SetFilter<T>(HashSet<T>);

impl<T: Eq + Hash> ENodeFilter<T> for SetFilter<T> {
    fn filter(&self, enodes: &mut Vec<T>) {
        enodes.retain(|node| self.0.contains(node))
    }
}

/// A summary of the cost of a node or class.
///
/// CostSummaries are used to customize the "update rule" for greedy extraction
/// algorithms.
pub(crate) trait CostSummary<NodeId>: Default {
    fn cost(&self) -> Cost;
    fn min(&self, other: &Self) -> Self;
    fn singleton(node: NodeId, cost: Cost) -> Self;
    fn combine(&mut self, other: &Self);
}

#[derive(Default)]
pub(crate) struct TreeCost(Option<NotNan<f64>>);

impl<NodeId> CostSummary<NodeId> for TreeCost {
    fn cost(&self) -> Cost {
        self.0.unwrap_or_else(|| Cost::new(f64::INFINITY).unwrap())
    }

    fn min(&self, other: &TreeCost) -> Self {
        match (self.0, other.0) {
            (None, None) => TreeCost::default(),
            (None, Some(x)) | (Some(x), None) => TreeCost(Some(x)),
            (Some(x), Some(y)) => TreeCost(Some(cmp::min(x, y))),
        }
    }

    fn singleton(_: NodeId, cost: Cost) -> Self {
        TreeCost(Some(cost))
    }

    fn combine(&mut self, other: &Self) {
        match (&mut self.0, other.0) {
            (None, _) => {}
            (_, None) => *self = Self::default(),
            (Some(x), Some(y)) => *x += y,
        }
    }
}
