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

pub(crate) type AnalysisResult<'a, E, C> = HashMap<
    <E as Egraph>::EClassId,
    NodeState<<E as Egraph>::ENodeId, Wrapper<C>, FixedPointSolver<'a, E, C>>,
>;

pub(crate) struct FixedPointSolver<'a, E: Egraph, C: CostSummary<Self, E::ENodeId>> {
    class_guesses: HashMap<E::EClassId, NodeState<E::ENodeId, Wrapper<C>, Self>>,
    costs: Rc<HashMap<E::ENodeId, Cost>>,
    egraph: &'a mut E,
}

impl<'a, E: Egraph, C: CostSummary<Self, E::ENodeId> + Clone> FixedPointSolver<'a, E, C> {
    pub(crate) fn new(
        egraph: &'a mut E,
        filter: &impl ENodeFilter<E::ENodeId>,
    ) -> FixedPointSolver<'a, E, C> {
        let mut class_guesses = HashMap::default();
        egraph.for_each_class(|class| {
            class_guesses.insert(
                class.clone(),
                NodeState {
                    node_id: None,
                    cost: Wrapper::default(),
                    _marker: PhantomData,
                },
            );
        });
        let mut costs = HashMap::default();
        let mut node_vec = Vec::new();
        for (class, _) in class_guesses.iter() {
            egraph.expand_class(class, &mut node_vec);
            for node in node_vec.drain(..) {
                let cost = egraph.cost(&node);
                costs.insert(node, cost);
            }
        }
        let mut res = FixedPointSolver {
            class_guesses,
            egraph,
            costs: Rc::new(costs),
        };
        res.iterate(filter);
        res
    }

    pub(crate) fn results(&self) -> &AnalysisResult<'a, E, C> {
        &self.class_guesses
    }

    fn iterate(&mut self, filter: &impl ENodeFilter<E::ENodeId>) {
        let pool = Pool::<E>::default();
        let mut classes = pool.class_vec();
        classes.extend(self.class_guesses.keys().cloned());
        let mut changed = true;
        while changed {
            changed = false;
            for class in classes.iter() {
                changed |= self.update_class(class, &pool, filter);
            }
        }
    }

    fn compute_cost(&mut self, node: &E::ENodeId, pool: &Pool<E>) -> Wrapper<C> {
        let mut classes = pool.class_vec();
        self.egraph.get_children(node, &mut classes);
        let mut cost = Wrapper::<C>::singleton(self, node.clone(), self.egraph.cost(node));
        for class in classes.drain(..) {
            cost.combine(&self.class_guesses[&class].cost)
        }
        cost
    }

    fn update_class(
        &mut self,
        class: &E::EClassId,
        pool: &Pool<E>,
        filter: &impl ENodeFilter<E::ENodeId>,
    ) -> bool {
        let mut changed = false;
        let mut nodes = pool.node_vec();
        self.egraph.expand_class(class, &mut nodes);
        filter.filter(&mut nodes);
        let mut cur = self.class_guesses[class].clone();
        for node in nodes.drain(..) {
            let cost = self.compute_cost(&node, pool);
            changed |= cur.update(&node, &cost);
        }

        if changed {
            self.class_guesses.insert(class.clone(), cur);
        }

        changed
    }
}

pub(crate) struct NodeState<N, C, S> {
    pub(crate) node_id: Option<N>,
    pub(crate) cost: C,
    _marker: PhantomData<*const S>,
}

impl<S, N: Clone, C: Clone> Clone for NodeState<N, C, S> {
    fn clone(&self) -> Self {
        NodeState {
            node_id: self.node_id.clone(),
            cost: self.cost.clone(),
            _marker: PhantomData,
        }
    }
}

impl<S, N: Clone, C: CostSummary<S, N>> NodeState<N, C, S> {
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

impl<S, N, C: CostSummary<S, N> + Clone> CostSummary<S, N> for Wrapper<C> {
    fn combine(&mut self, other: &Self) {
        match (&mut self.0, &other.0) {
            (None, _) => {}
            (_, None) => *self = Self::default(),
            (Some(x), Some(y)) => x.combine(y),
        }
    }

    fn cost(&self) -> Cost {
        self.0
            .as_ref()
            .map(|x| x.cost())
            .unwrap_or_else(|| Cost::new(f64::INFINITY).unwrap())
    }

    fn min(&self, other: &Self) -> Self {
        match (&self.0, &other.0) {
            (None, None) => Wrapper::default(),
            (None, Some(x)) | (Some(x), None) => Wrapper(Some(x.clone())),
            (Some(x), Some(y)) => Wrapper(Some(x.min(y))),
        }
    }

    fn singleton(state: &S, node: N, cost: Cost) -> Self {
        Wrapper(Some(C::singleton(state, node, cost)))
    }
}

// TODO: lots to update here with CostSummary:
// * cost can probably just go away, in favor of a "State" arg, which is
//   actually just passed in to the constructor. State can be an associated
//   type.
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
pub(crate) trait CostSummary<State, NodeId>: Sized {
    fn cost(&self) -> Cost;
    fn min(&self, other: &Self) -> Self;
    fn min_update(&mut self, other: &Self) -> bool {
        let res = self.min(other);
        debug_assert!(res.cost() <= self.cost());
        if res.cost() < self.cost() {
            *self = res;
            true
        } else {
            false
        }
    }
    fn singleton(state: &State, node: NodeId, cost: Cost) -> Self;
    fn combine(&mut self, other: &Self);
}

#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct TreeCost(Cost);

impl<State, NodeId> CostSummary<State, NodeId> for TreeCost {
    fn cost(&self) -> Cost {
        self.0
    }

    fn min(&self, other: &TreeCost) -> Self {
        TreeCost(cmp::min(self.0, other.0))
    }

    fn singleton(_: &State, _: NodeId, cost: Cost) -> Self {
        TreeCost(cost)
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

impl<'a, E: Egraph + 'a> CostSummary<FixedPointSolver<'a, E, Self>, E::ENodeId>
    for DagCost<E::ENodeId>
{
    fn combine(&mut self, other: &Self) {
        self.set.union_agg(&other.set, |n| AddCost(self.costs[n]))
    }

    fn cost(&self) -> Cost {
        self.set.agg().0
    }

    fn min(&self, other: &Self) -> Self {
        if CostSummary::<FixedPointSolver<'a, E, Self>, E::ENodeId>::cost(self)
            <= CostSummary::<FixedPointSolver<'a, E, Self>, E::ENodeId>::cost(other)
        {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn singleton(state: &FixedPointSolver<'a, E, Self>, node: E::ENodeId, _: Cost) -> Self {
        let mut set = val_trie::HashSet::<E::ENodeId, AddCost>::default();
        set.insert_agg(node, |n| AddCost(state.costs[n]));
        DagCost {
            set,
            costs: state.costs.clone(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct MentionedSet<T> {
    set: val_trie::HashSet<T>,
}

impl<'a, E: Egraph + 'a> CostSummary<FixedPointSolver<'a, E, Self>, E::ENodeId>
    for MentionedSet<E::ENodeId>
{
    fn combine(&mut self, other: &Self) {
        self.set.union(&other.set)
    }

    fn cost(&self) -> Cost {
        Cost::default()
    }

    fn min(&self, other: &Self) -> Self {
        let mut set = self.set.clone();
        set.union(&other.set);
        MentionedSet { set }
    }

    fn singleton(_: &FixedPointSolver<'a, E, Self>, n: E::ENodeId, _: Cost) -> Self {
        let mut set = val_trie::HashSet::<E::ENodeId>::default();
        set.insert(n);
        MentionedSet { set }
    }
}

struct ZddNode(u32);

// impl<'a, E:Egraph+'a> CostSummary<FixedPointSolver<'a, E, Self>, E::ENodeId> for Zdd<>
