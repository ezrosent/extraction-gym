//! A simple associative caching scheme for operations on a ZDD.
//!
//! Knuth describes this strategy in TAOCP 4A in the section on BDDs.

use std::{
    cmp,
    hash::{Hash, Hasher},
    mem,
};

use rustc_hash::FxHasher;

pub(crate) struct Cache<K, V> {
    max_size: usize,
    populated: usize,
    hits: usize,
    misses: usize,
    table: Vec<Option<(K, V)>>,
}

impl<K: Hash + Eq, V> Cache<K, V> {
    /// Create a new cache with size bounded by `cap`.
    pub(crate) fn new(cap: usize) -> Cache<K, V> {
        let mut table = Vec::new();
        let cap = cap.next_power_of_two();
        table.resize_with(cmp::min(128, cap), || None);
        Cache {
            max_size: cap,
            populated: 0,
            hits: 0,
            misses: 0,
            table,
        }
    }

    /// Take the current entries in the cache and rewrite them according to `f`
    /// in place. If the function returns 'false' then the item should be
    /// removed.
    pub(crate) fn remap(&mut self, mut f: impl FnMut(&mut K, &mut V) -> bool) {
        for i in 0..self.table.len() {
            let slot = &mut self.table[i];
            if slot.is_none() {
                continue;
            }
            let (mut k, mut v) = slot.take().unwrap();
            self.populated -= 1;
            if f(&mut k, &mut v) {
                self.set_inner(k, v)
            }
        }
    }

    fn get_index(&self, k: &K) -> usize {
        let mut hasher = FxHasher::default();
        k.hash(&mut hasher);
        (hasher.finish() as usize) & (self.table.len() - 1)
    }

    fn should_grow(&self) -> bool {
        self.max_size > self.table.len() && ((self.table.len() * 3) / 4) <= self.populated
    }

    fn maybe_grow(&mut self) {
        if !self.should_grow() {
            return;
        }
        let mut new_vec = Vec::new();
        new_vec.resize_with(self.table.len() * 2, || None);
        mem::swap(&mut new_vec, &mut self.table);
        self.populated = 0;
        for (k, v) in new_vec.into_iter().flatten() {
            self.set(k, v);
        }
    }

    /// The ratio of cache hits to the total number of calls to `get`.
    pub(crate) fn hit_ratio(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses) as f64
    }

    /// The number of slots currently populated in memory.
    ///
    /// The table can grow to the maximum capacity provided in `new`, but it
    /// only allocates that space lazily.
    pub(crate) fn capacity(&self) -> usize {
        self.table.len()
    }

    /// Get a cached value, if one is present.
    pub(crate) fn get(&mut self, k: &K) -> Option<&V> {
        let (candidate, v) = if let Some(x) = self.table[self.get_index(k)].as_ref() {
            x
        } else {
            self.misses += 1;
            return None;
        };
        if k == candidate {
            self.hits += 1;
            Some(v)
        } else {
            self.misses += 1;
            None
        }
    }

    fn set_inner(&mut self, k: K, v: V) {
        let ix = self.get_index(&k);
        let slot = &mut self.table[ix];
        self.populated += usize::from(slot.is_none());
        *slot = Some((k, v));
    }

    /// Map `k` to `v` in the cache.
    pub(crate) fn set(&mut self, k: K, v: V) {
        self.set_inner(k, v);
        self.maybe_grow();
    }
}
