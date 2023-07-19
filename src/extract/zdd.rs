use std::collections::HashMap;

use egraph_serialize::{ClassId, NodeId};

use crate::{Cost, ExtractionResult, Extractor};

pub struct ZddExtractor {
    pub node_limit: usize,
}

impl Extractor for ZddExtractor {
    fn extract(&self, egraph: &egraph_serialize::EGraph, roots: &[ClassId]) -> ExtractionResult {
        assert_eq!(roots.len(), 1);
        let root = &roots[0];
        let (mut flat, flat_root) = FlatEgraph::new(egraph, root);

        let res = zdds::extract_zdd(&mut flat, flat_root, Some(self.node_limit))
            .expect("extraction failed");
        flat.convert_extract_result(&res)
    }
}

type Id = usize;

struct Node {
    id: NodeId,
    cost: Cost,
    children: Vec<Id>,
}

struct Class {
    id: ClassId,
    nodes: Vec<Id>,
}

struct FlatEgraph<'a> {
    nodes: Vec<Node>,
    classes: Vec<Class>,
    src: &'a egraph_serialize::EGraph,
}

impl<'a> FlatEgraph<'a> {
    fn new(eg: &'a egraph_serialize::EGraph, root: &ClassId) -> (Self, Id) {
        let mut root_id = !0usize;
        let mut nodes = Vec::new();
        let mut classes = Vec::new();
        let mut node_ids = HashMap::new();
        let mut class_ids = HashMap::new();
        for (id, node) in eg.nodes.iter() {
            node_ids.insert(id.clone(), nodes.len());
            nodes.push(Node {
                id: id.clone(),
                cost: node.cost,
                children: Vec::new(),
            });
        }
        for (id, _) in eg.classes().iter() {
            class_ids.insert(id.clone(), classes.len());
            if id == root {
                root_id = classes.len();
            }
            classes.push(Class {
                id: id.clone(),
                nodes: Vec::new(),
            });
        }

        for ((_, src), dst) in eg.nodes.iter().zip(nodes.iter_mut()) {
            for child in &src.children {
                dst.children.push(class_ids[eg.nid_to_cid(child)])
            }
        }

        for ((_, src), dst) in eg.classes().iter().zip(classes.iter_mut()) {
            for node in &src.nodes {
                dst.nodes.push(node_ids[node]);
            }
        }

        (
            FlatEgraph {
                nodes,
                classes,
                src: eg,
            },
            root_id,
        )
    }

    fn convert_extract_result(&self, res: &zdds::ExtractResult<Id>) -> ExtractionResult {
        let mut converted = ExtractionResult::default();
        for node_id in res.dag.node_weights() {
            let node = &self.nodes[*node_id];
            let class_id = self.src.nid_to_cid(&node.id);
            converted.choose(class_id.clone(), node.id.clone());
        }

        converted
    }
}

impl<'a> zdds::Egraph for FlatEgraph<'a> {
    type EClassId = Id;
    type ENodeId = Id;

    fn cost(&self, node: &Id) -> Cost {
        self.nodes[*node].cost
    }

    fn expand_class(&mut self, class: &Id, nodes: &mut Vec<Self::ENodeId>) {
        nodes.extend_from_slice(&self.classes[*class].nodes)
    }

    fn get_children(&mut self, node: &Id, classes: &mut Vec<Self::EClassId>) {
        classes.extend_from_slice(&self.nodes[*node].children)
    }

    fn try_for_each_class<E>(
        &self,
        mut f: impl FnMut(&Self::EClassId) -> Result<(), E>,
    ) -> Result<(), E> {
        for class in 0..self.classes.len() {
            f(&class)?;
        }
        Ok(())
    }
}
