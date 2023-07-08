//! An abstraction for Egraphs and utilities for building ZDDs to represent
//! possible choices of ENodes.
use std::{
    cell::RefCell,
    hash::{BuildHasherDefault, Hash},
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

use hashbrown::{hash_map::Entry, HashMap};
use indexmap::IndexSet;
use ordered_float::NotNan;
use rustc_hash::FxHasher;

use crate::{cycle_patch::CyclePatcher, extract::greedy_costs, zdd::NodeId, HashSet, Zdd, ZddPool};

pub(crate) type Cost = NotNan<f64>;

/// The `Egraph` trait encapsulates the core information required from an Egraph
/// to encode the extraction problem.
pub trait Egraph {
    type EClassId: Eq + Clone + Hash + Ord;
    type ENodeId: Eq + Clone + Hash;
    // Instead of returning into a vector, it'd be nice to return impl
    // Iterator<...>, but that is not stable yet.

    /// Optional printing routine: just for debugging purposes.
    fn print_node(&mut self, _node: &Self::ENodeId) -> String {
        Default::default()
    }

    fn try_for_each_class<E>(
        &self,
        f: impl FnMut(&Self::EClassId) -> Result<(), E>,
    ) -> Result<(), E>;

    fn for_each_class(&self, mut f: impl FnMut(&Self::EClassId)) {
        let _ = self.try_for_each_class::<()>(|c| {
            f(c);
            Ok(())
        });
    }

    /// For a given EClass, push all of its component ENodes into the `nodes` vector.
    fn expand_class(&mut self, class: &Self::EClassId, nodes: &mut Vec<Self::ENodeId>);
    /// For a given ENode, push all of its children into the `classes` vector.
    fn get_children(&mut self, node: &Self::ENodeId, classes: &mut Vec<Self::EClassId>);

    fn cost(&self, node: &Self::ENodeId) -> Cost;
}

/// Create a mermaid diagram of the ZDD representing possible extractions from
/// the egraph. This function does not pass a node limit.
pub fn render_zdd<E: Egraph>(
    egraph: &mut E,
    root: E::EClassId,
    mut render_node: impl FnMut(&E::ENodeId) -> String,
) -> String {
    let extractor = PureZddExtractor::new(root, egraph, None);
    if let Some(zdd) = &extractor.zdd {
        zdd.mermaid_diagram(|zdd_node| {
            render_node(
                extractor
                    .node_mapping
                    .get_index(zdd_node.index())
                    .expect("all nodes should be valid"),
            )
        })
    } else {
        "extraction failed".into()
    }
}

/// Given an Egraph, pick the minimum-cost set of enodes to be used during
/// extraction.
pub fn choose_nodes<E: Egraph>(
    egraph: &mut E,
    root: E::EClassId,
    node_limit: Option<usize>,
) -> Option<(Vec<E::ENodeId>, Cost)> {
    // First, run the greedy algorithm to get an estimate of the cost of e-graphs "under" a cycle.
    // TODO: we could lazily evaluate this computation.
    let greedy_extractor = greedy_costs(egraph, &root);

    // Then, initialize a worklist of classes to build ZDDs for.
    let mut patcher = CyclePatcher::new(root.clone());
    let mut zdd_extractor =
        PureZddExtractor::with_pool(node_limit, ZddPool::with_cache_size(1 << 30));
    let pool = Pool::default();

    // For each class that is either the root or is transitively mentioned in a
    // cycle reachable from the root:
    while let Some(class) = patcher.next_node() {
        // Build a ZDD and extract the minimum-cost set, slotting in greedy costs for "placeholders".
        let zdd = zdd_extractor.solve_class(class.clone(), egraph, &pool);
        let (zdd_nodes, _) = zdd.min_cost_set(|zdd_node| match zdd_node {
            ZddNode::ENode(n) => {
                let node = zdd_extractor
                    .node_mapping
                    .get_index(*n as usize)
                    .expect("all nodes should be valid");
                egraph.cost(node)
            }
            ZddNode::Placeholder(c) => greedy_extractor
                .get(c)
                .cloned()
                .unwrap_or(Cost::new(f64::INFINITY).unwrap()),
        })?;
        // For non-placeholder nodes in the set, register them as "elements".
        // Register any placeholders as dependencies.
        for node in zdd_nodes {
            match node {
                ZddNode::ENode(n) => patcher.add_elt(class.clone(), n),
                ZddNode::Placeholder(dep) => patcher.add_dep(class.clone(), dep),
            }
        }
        log::debug!("finished minor iteration");
    }

    // Iterate the patcher to a fixed point. Any if we called add_dep(c1, c2),
    // then any nodes added to c2 will be added to c1 as well.
    patcher.iterate();

    // Read the final set out of the patcher.
    let mut nodes = Vec::new();
    let mut cost = Cost::default();
    for i in patcher.elts(&root) {
        let node = zdd_extractor
            .node_mapping
            .get_index(*i as usize)
            .expect("all nodes should be valid");
        nodes.push(node.clone());
        cost += egraph.cost(node);
    }
    Some((nodes, cost))
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum ZddNode<T> {
    ENode(u32),
    Placeholder(T),
}

impl<T> ZddNode<T> {
    fn new(u: usize) -> ZddNode<T> {
        assert!(u <= u32::MAX as usize);
        ZddNode::ENode(u as u32)
    }

    fn placeholder(t: T) -> ZddNode<T> {
        ZddNode::Placeholder(t)
    }

    fn index(&self) -> usize {
        match self {
            ZddNode::ENode(n) => *n as usize,
            ZddNode::Placeholder(_) => panic!("tried to get index of placeholder"),
        }
    }
}

pub(crate) struct PureZddExtractor<E: Egraph> {
    /// Assigns each e-node to its offset in the IndexSet. We want to assign our
    /// own offsets because (heuristically) nodes given a topological order will
    /// compress better in the ZDD.
    node_mapping: IndexSet<E::ENodeId, BuildHasherDefault<FxHasher>>,
    /// The ZDD encoding all the possible choices of ENode.
    zdd: Option<Zdd<ZddNode<E::EClassId>>>,
    bot: Zdd<ZddNode<E::EClassId>>,
    visited_set: HashSet<NodeId>,
    node_limit: usize,
}

impl<E: Egraph> PureZddExtractor<E> {
    fn with_pool(
        node_limit: Option<usize>,
        pool: ZddPool<ZddNode<E::EClassId>>,
    ) -> PureZddExtractor<E> {
        let node_mapping = IndexSet::default();
        let bot = Zdd::with_pool(pool);
        PureZddExtractor {
            node_mapping,
            zdd: None,
            bot,
            visited_set: Default::default(),
            node_limit: node_limit.unwrap_or(usize::MAX),
        }
    }

    pub(crate) fn new(
        root: E::EClassId,
        egraph: &mut E,
        node_limit: Option<usize>,
    ) -> PureZddExtractor<E> {
        let pool = Pool::default();
        let mut res = Self::with_pool(node_limit, ZddPool::with_cache_size(1 << 25));
        let root = res.solve_class(root, egraph, &pool);
        res.zdd = Some(root);
        res
    }

    fn solve_class(
        &mut self,
        class: E::EClassId,
        egraph: &mut E,
        pool: &Pool<E>,
    ) -> Zdd<ZddNode<E::EClassId>> {
        self.traverse_class(&mut Default::default(), &class, egraph, pool)
    }

    fn traverse_class(
        &mut self,
        visited: &mut VisitedSet<E::EClassId>,
        class: &E::EClassId,
        egraph: &mut E,
        pool: &Pool<E>,
    ) -> Zdd<ZddNode<E::EClassId>> {
        match visited.entry(class.clone()) {
            Entry::Occupied(o) => {
                return match o.get() {
                    Some(n) => n.clone(),
                    None => {
                        Zdd::singleton(self.bot.pool().clone(), ZddNode::placeholder(class.clone()))
                    }
                }
            }
            Entry::Vacant(v) => v.insert(None),
        };
        let mut nodes = pool.node_vec();
        egraph.expand_class(class, &mut nodes);
        let start_nodes = self.bot.pool().size();
        let start_gcs = self.bot.pool().num_gcs();
        let mut outer_nodes = pool.zdd_vec();
        for node in nodes.drain(..) {
            outer_nodes.push(self.traverse_node(visited, node, egraph, pool));
        }
        if outer_nodes.is_empty() {
            return self.bot.clone();
        }
        let mut composite = outer_nodes.pop().unwrap();
        for node in outer_nodes.drain(..) {
            composite.merge(&node);
        }
        let pool_delta = self.bot.pool().size().saturating_sub(start_nodes);
        let end_gcs = self.bot.pool().num_gcs();
        if (end_gcs > start_gcs || pool_delta > self.node_limit)
            && composite.count_nodes(&mut self.visited_set) > self.node_limit
        {
            composite.freeze();
        }

        visited.insert(class.clone(), Some(composite.clone()));
        composite
    }

    fn traverse_node(
        &mut self,
        visited: &mut VisitedSet<E::EClassId>,
        node: E::ENodeId,
        egraph: &mut E,
        pool: &Pool<E>,
    ) -> Zdd<ZddNode<E::EClassId>> {
        let node_id = self.get_zdd_node(&node);

        let mut classes = pool.class_vec();
        egraph.get_children(&node, &mut classes);
        let start_nodes = self.bot.pool().size();
        let start_gcs = self.bot.pool().num_gcs();
        let mut inner_nodes = pool.zdd_vec();
        for class in classes.drain(..) {
            inner_nodes.push(self.traverse_class(visited, &class, egraph, pool));
        }

        let mut composite = Zdd::singleton(self.bot.pool().clone(), node_id);
        for node in inner_nodes.drain(..) {
            composite.join(&node);
        }

        let pool_delta = self.bot.pool().size().saturating_sub(start_nodes);
        let end_gcs = self.bot.pool().num_gcs();
        if (end_gcs > start_gcs || pool_delta > self.node_limit)
            && composite.count_nodes(&mut self.visited_set) > self.node_limit
        {
            composite.freeze();
        }

        composite
    }

    fn get_zdd_node(&mut self, node: &E::ENodeId) -> ZddNode<E::EClassId> {
        if let Some(x) = self.node_mapping.get_index_of(node) {
            ZddNode::new(x)
        } else {
            let (offset, _) = self.node_mapping.insert_full(node.clone());
            ZddNode::new(offset)
        }
    }
}

type VisitedSet<T> = HashMap<T, Option<Zdd<ZddNode<T>>>>;

type ZddVec2<T> = Vec<Vec<Zdd<ZddNode<T>>>>;

pub(crate) struct Pool<E: Egraph> {
    classes: RefCell<Vec<Vec<E::EClassId>>>,
    nodes: RefCell<Vec<Vec<E::ENodeId>>>,
    zdds: RefCell<ZddVec2<E::EClassId>>,
}

impl<E: Egraph> Default for Pool<E> {
    fn default() -> Pool<E> {
        Pool {
            classes: Default::default(),
            nodes: Default::default(),
            zdds: Default::default(),
        }
    }
}

impl<E: Egraph> Pool<E> {
    pub(crate) fn class_vec(&self) -> PoolRef<Vec<E::EClassId>> {
        let res = self.classes.borrow_mut().pop().unwrap_or_default();
        PoolRef {
            elt: ManuallyDrop::new(res),
            pool: &self.classes,
        }
    }
    pub(crate) fn node_vec(&self) -> PoolRef<Vec<E::ENodeId>> {
        let res = self.nodes.borrow_mut().pop().unwrap_or_default();
        PoolRef {
            elt: ManuallyDrop::new(res),
            pool: &self.nodes,
        }
    }
    fn zdd_vec(&self) -> PoolRef<Vec<Zdd<ZddNode<E::EClassId>>>> {
        let res = self.zdds.borrow_mut().pop().unwrap_or_default();
        PoolRef {
            elt: ManuallyDrop::new(res),
            pool: &self.zdds,
        }
    }
}

pub(crate) trait Clearable {
    fn clear(&mut self);
}

impl<T> Clearable for Vec<T> {
    fn clear(&mut self) {
        self.clear()
    }
}

pub(crate) struct PoolRef<'a, T: Clearable> {
    elt: ManuallyDrop<T>,
    pool: &'a RefCell<Vec<T>>,
}

impl<'a, T: Clearable> Deref for PoolRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.elt
    }
}

impl<'a, T: Clearable> DerefMut for PoolRef<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.elt
    }
}

impl<'a, T: Clearable> Drop for PoolRef<'a, T> {
    fn drop(&mut self) {
        self.elt.clear();
        let t = (&self.elt) as *const ManuallyDrop<T>;
        let elt = unsafe { ManuallyDrop::into_inner(t.read()) };
        self.pool.borrow_mut().push(elt);
    }
}
