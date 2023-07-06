use std::collections::HashMap;

pub use crate::*;

pub mod bottom_up;

#[cfg(feature = "ilp-cbc")]
pub mod ilp_cbc;

pub trait Extractor: Sync {
    fn extract(&self, egraph: &SimpleEGraph, roots: &[Id]) -> ExtractionResult;

    fn boxed(self) -> Box<dyn Extractor>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

#[derive(Clone)]
pub struct ExtractionResult {
    pub choices: Vec<Id>,
    tree_memo: HashMap<Id, Cost>,
}

impl ExtractionResult {
    pub fn new(n_classes: usize) -> Self {
        ExtractionResult {
            choices: vec![0; n_classes],
            tree_memo: HashMap::default(),
        }
    }

    pub fn tree_cost(&mut self, egraph: &SimpleEGraph, root: Id) -> Cost {
        if let Some(&cost) = self.tree_memo.get(&root) {
            return cost;
        }
        let node = &egraph[root].nodes[self.choices[root]];
        let mut cost = node.cost;
        for &child in &node.children {
            cost += self.tree_cost(egraph, child);
        }
        self.tree_memo.insert(root, cost);
        cost
    }

    // this will loop if there are cycles
    pub fn dag_cost(&self, egraph: &SimpleEGraph, root: Id) -> Cost {
        let mut costs = vec![INFINITY; egraph.classes.len()];
        let mut todo = vec![root];
        while let Some(i) = todo.pop() {
            let node = &egraph[i].nodes[self.choices[i]];
            costs[i] = node.cost;
            for &child in &node.children {
                if costs[child] == INFINITY {
                    todo.push(child);
                }
            }
        }
        costs.iter().filter(|c| **c != INFINITY).sum()
    }

    pub fn node_sum_cost(&self, node: &Node, costs: &[Cost]) -> Cost {
        node.cost + node.children.iter().map(|&i| costs[i]).sum::<Cost>()
    }
}
