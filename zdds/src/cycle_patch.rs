//! Utilities for adding back the content of ZDD extractions "hidden" by cycles.

use std::{hash::Hash, mem};

use indexmap::IndexSet;
use petgraph::{graph::NodeIndex, Graph};

use crate::{HashMap, HashSet};

struct NodeState<T> {
    // TODO: look into val-trie here
    elts: HashSet<T>,
    node: NodeIndex,
    visited: bool,
}

pub(crate) struct CyclePatcher<C, N> {
    to_visit: IndexSet<C>,
    visited: HashMap<C, NodeState<N>>,
    deps: Graph<C, ()>,
}

impl<C, N> Default for CyclePatcher<C, N> {
    fn default() -> Self {
        Self {
            to_visit: Default::default(),
            visited: Default::default(),
            deps: Default::default(),
        }
    }
}

impl<C: Hash + Eq + Clone, N: Hash + Eq + Clone> CyclePatcher<C, N> {
    pub(crate) fn new(class: C) -> Self {
        let mut this = Self::default();
        this.to_visit.insert(class);
        this
    }
    pub(crate) fn elts(&self, class: &C) -> impl Iterator<Item = &N> {
        self.visited
            .get(class)
            .into_iter()
            .flat_map(|state| state.elts.iter())
    }

    pub(crate) fn add_elt(&mut self, k: C, v: N) {
        self.to_visit.remove(&k);
        let state = self.get_state(k, true);
        state.elts.insert(v);
    }

    pub(crate) fn add_dep(&mut self, c1: C, c2: C) {
        let state = self.get_state(c2.clone(), false);
        let src_node = state.node;

        if !state.visited {
            self.to_visit.insert(c2);
        }

        let state = self.get_state(c1, true);
        let target_node = state.node;

        self.deps.add_edge(src_node, target_node, ());
    }

    pub(crate) fn next_node(&mut self) -> Option<C> {
        let res = self.to_visit.pop()?;
        self.get_state(res.clone(), true).visited = true;
        Some(res)
    }

    pub(crate) fn iterate(&mut self) {
        let mut worklist: IndexSet<NodeIndex> = self.deps.node_indices().collect();
        while let Some(ix) = worklist.pop() {
            let src = self.deps.node_weight(ix).unwrap();
            for dst_ix in self.deps.neighbors(ix) {
                let dst = self.deps.node_weight(dst_ix).unwrap();
                let mut elts = mem::take(&mut self.visited.get_mut(dst).unwrap().elts);
                let start_len = elts.len();
                elts.extend(self.visited[src].elts.iter().cloned());
                let delta = elts.len() - start_len;
                self.visited.get_mut(dst).unwrap().elts = elts;

                if delta > 0 {
                    worklist.insert(dst_ix);
                }
            }
        }
    }

    fn get_state(&mut self, class: C, start_visited: bool) -> &mut NodeState<N> {
        let graph = &mut self.deps;
        self.visited
            .entry(class.clone())
            .or_insert_with(|| NodeState {
                elts: HashSet::default(),
                node: graph.add_node(class.clone()),
                visited: start_visited,
            })
    }
}
