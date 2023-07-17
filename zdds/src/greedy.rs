//! Greedy Extraction.

use std::{cell::RefCell, cmp, hash::Hash, marker::PhantomData, rc::Rc};

use crate::{
    egraph::{Cost, Pool},
    extract::ENodeFilter,
    Egraph, HashMap, Zdd,
};

// TODO: remove extraction results
// TODO: remove existing greedy implementation

type GreedyTree<'a, E> = FixedPointSolver<'a, E, TreeCost>;
// GreedyExtractor will check the actual bound.
#[allow(type_alias_bounds)]
type GreedyDag<'a, E: Egraph> = FixedPointSolver<'a, E, DagCost<<E as Egraph>::ENodeId>>;

pub(crate) type AnalysisResult<'a, E, C> =
    HashMap<<E as Egraph>::EClassId, NodeState<<E as Egraph>::ENodeId, Wrapper<C>>>;

pub(crate) struct FixedPointSolver<'a, E: Egraph, C: CostSummary<E::ENodeId>> {
    class_guesses: HashMap<E::EClassId, NodeState<E::ENodeId, Wrapper<C>>>,
    egraph: &'a mut E,
}

fn compute_costs<E: Egraph>(egraph: &mut E) -> HashMap<E::ENodeId, Cost> {
    let mut costs = HashMap::default();
    let mut node_vec = Vec::new();
    let mut classes = Vec::new();
    egraph.for_each_class(|x| classes.push(x.clone()));
    for class in classes {
        egraph.expand_class(&class, &mut node_vec);
        for node in node_vec.drain(..) {
            let cost = egraph.cost(&node);
            costs.insert(node, cost);
        }
    }
    costs
}

impl<'a, E: Egraph, C: CostSummary<E::ENodeId> + Clone + Eq> FixedPointSolver<'a, E, C> {
    pub(crate) fn new(
        egraph: &'a mut E,
        filter: &impl ENodeFilter<E::ENodeId>,
        state: &mut C::State,
    ) -> FixedPointSolver<'a, E, C> {
        let mut class_guesses = HashMap::default();
        egraph.for_each_class(|class| {
            class_guesses.insert(
                class.clone(),
                NodeState {
                    node_id: None,
                    cost: Wrapper::default(),
                },
            );
        });

        let mut res = FixedPointSolver {
            class_guesses,
            egraph,
        };
        res.iterate(state, filter);
        res
    }

    pub(crate) fn results(&self) -> &AnalysisResult<'a, E, C> {
        &self.class_guesses
    }

    fn iterate(&mut self, state: &mut C::State, filter: &impl ENodeFilter<E::ENodeId>) {
        let pool = Pool::<E>::default();
        let mut classes = pool.class_vec();
        classes.extend(self.class_guesses.keys().cloned());
        let mut changed = true;
        while changed {
            changed = false;
            for class in classes.iter() {
                changed |= self.update_class(class, &pool, state, filter);
            }
        }
    }

    fn compute_cost(
        &mut self,
        node: &E::ENodeId,
        pool: &Pool<E>,
        state: &mut C::State,
    ) -> Wrapper<C> {
        let mut classes = pool.class_vec();
        self.egraph.get_children(node, &mut classes);
        let mut cost = Wrapper::<C>::singleton(state, node.clone());
        for class in classes.drain(..) {
            cost.combine(&self.class_guesses[&class].cost)
        }
        cost
    }

    fn update_class(
        &mut self,
        class: &E::EClassId,
        pool: &Pool<E>,
        state: &mut C::State,
        filter: &impl ENodeFilter<E::ENodeId>,
    ) -> bool {
        let mut changed = false;
        let mut nodes = pool.node_vec();
        self.egraph.expand_class(class, &mut nodes);
        filter.filter(&mut nodes);
        let mut cur = self.class_guesses[class].clone();
        for node in nodes.drain(..) {
            let cost = self.compute_cost(&node, pool, state);
            changed |= cur.update(&node, &cost);
        }

        if changed {
            self.class_guesses.insert(class.clone(), cur);
        }

        changed
    }
}

pub(crate) struct NodeState<N, C> {
    pub(crate) node_id: Option<N>,
    pub(crate) cost: C,
}

impl<N: Clone, C: Clone> Clone for NodeState<N, C> {
    fn clone(&self) -> Self {
        NodeState {
            node_id: self.node_id.clone(),
            cost: self.cost.clone(),
        }
    }
}

impl<N: Clone, C: CostSummary<N> + Eq> NodeState<N, C> {
    fn update(&mut self, node: &N, c: &C) -> bool {
        if self.cost.min_update(c) {
            self.node_id = Some(node.clone());
            true
        } else {
            false
        }
    }
}

#[derive(PartialEq, Eq, Clone)]
pub(crate) struct Wrapper<C>(Option<C>);

impl<C> Wrapper<C> {
    pub(crate) fn cost(&self) -> Option<&C> {
        self.0.as_ref()
    }
}

impl<C> Default for Wrapper<C> {
    fn default() -> Self {
        Wrapper(None)
    }
}

impl<N, C: CostSummary<N> + Clone> CostSummary<N> for Wrapper<C> {
    type State = C::State;
    fn combine(&mut self, other: &Self) {
        match (&mut self.0, &other.0) {
            (None, _) => {}
            (_, None) => *self = Self::default(),
            (Some(x), Some(y)) => x.combine(y),
        }
    }

    fn min(&self, other: &Self) -> Self {
        match (&self.0, &other.0) {
            (None, None) => Wrapper::default(),
            (None, Some(x)) | (Some(x), None) => Wrapper(Some(x.clone())),
            (Some(x), Some(y)) => Wrapper(Some(x.min(y))),
        }
    }

    fn singleton(state: &mut Self::State, node: N) -> Self {
        Wrapper(Some(C::singleton(state, node)))
    }
}

// TODO: lots to update here with CostSummary:
// * This State arg can be something like a ZddPool + intern table.
// * CostSummary should implement Eq and min_update should be a basic thing that
// does min, sees if it equals self, and if it doesn't, assigns and returns
// true.

// At that point it's a lot closer to a basic semiring with some extra state
// attached. An ENodeSemiring?

/// A summary of the cost of a node or class.
///
/// CostSummaries are used to customize the "update rule" for greedy extraction
/// algorithms.
pub(crate) trait CostSummary<NodeId>: Sized {
    type State;
    fn min(&self, other: &Self) -> Self;
    fn min_update(&mut self, other: &Self) -> bool
    where
        Self: Eq,
    {
        let res = self.min(other);
        if &res != self {
            *self = res;
            true
        } else {
            false
        }
    }
    fn singleton(state: &mut Self::State, node: NodeId) -> Self;
    fn combine(&mut self, other: &Self);
}

#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct TreeCost(Cost);

impl<NodeId: Hash + Eq> CostSummary<NodeId> for TreeCost {
    type State = HashMap<NodeId, Cost>;
    fn min(&self, other: &TreeCost) -> Self {
        TreeCost(cmp::min(self.0, other.0))
    }

    fn singleton(state: &mut Self::State, n: NodeId) -> Self {
        TreeCost(state[&n])
    }

    fn combine(&mut self, other: &Self) {
        self.0 += other.0;
    }
}

#[derive(Default, Clone, Copy)]
struct AddCost(Cost);

impl val_trie::Group for AddCost {
    fn add(&mut self, other: &Self) {
        self.0 += other.0;
    }

    fn inverse(&self) -> Self {
        AddCost(-self.0)
    }

    fn sub(&mut self, other: &Self) {
        self.0 -= other.0;
    }
}

pub(crate) struct DagCost<T> {
    set: val_trie::HashSet<T, AddCost>,
    costs: Rc<HashMap<T, Cost>>,
}

impl<T: Clone> Clone for DagCost<T> {
    fn clone(&self) -> Self {
        DagCost {
            set: self.set.clone(),
            costs: self.costs.clone(),
        }
    }
}

impl<T: Hash + Eq + Clone> PartialEq for DagCost<T> {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set && Rc::ptr_eq(&self.costs, &other.costs)
    }
}

impl<T: Hash + Eq + Clone> Eq for DagCost<T> {}

impl<Node: Hash + Eq + Clone> CostSummary<Node> for DagCost<Node> {
    type State = Rc<HashMap<Node, Cost>>;

    fn combine(&mut self, other: &Self) {
        self.set.union_agg(&other.set, |n| AddCost(self.costs[n]))
    }

    fn min(&self, other: &Self) -> Self {
        if self.set.agg().0 <= other.set.agg().0 {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn singleton(state: &mut Self::State, node: Node) -> Self {
        let mut set = val_trie::HashSet::<Node, AddCost>::default();
        set.insert_agg(node, |n| AddCost(state[n]));
        DagCost {
            set,
            costs: state.clone(),
        }
    }
}
