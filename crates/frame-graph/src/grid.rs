//! A texture's subresources, split only as finely as its uses need, for the
//! compiling steps that keep state per subresource: the dependency graph
//! and barrier derivation.

use std::ops::Range;

/// A texture's subresources, split only as finely as its uses need: its
/// mips at every bound a use's mips have, and its layers likewise, so a
/// texture whose size the graph doesn't know is split as one it does. Each
/// cell, a range of mips of a range of layers, holds a `T`.
#[derive(Debug, Clone)]
pub(crate) struct Grid<T> {
    /// The bounds the mips are split at, in order.
    mips: Vec<u32>,
    /// The bounds the layers are split at, in order.
    layers: Vec<u32>,
    /// Each cell's state, by mip interval then layer interval.
    pub(crate) cells: Vec<T>,
}

impl<T> Default for Grid<T> {
    /// A texture not yet split, with no cells.
    fn default() -> Self {
        Self {
            mips: Vec::new(),
            layers: Vec::new(),
            cells: Vec::new(),
        }
    }
}

impl<T: Clone + Default> Grid<T> {
    /// Splits the texture at the bounds of `mips` and `layers`.
    pub(crate) fn split(&mut self, mips: &Range<u32>, layers: &Range<u32>) {
        self.mips.extend([mips.start, mips.end]);
        self.layers.extend([layers.start, layers.end]);
    }

    /// The texture, split at every bound, with each cell's state the
    /// default.
    pub(crate) fn finish(mut self) -> Self {
        for bounds in [&mut self.mips, &mut self.layers] {
            bounds.sort_unstable();
            bounds.dedup();
        }
        let count = |bounds: &[u32]| bounds.len().saturating_sub(1);
        self.cells = vec![T::default(); count(&self.mips) * count(&self.layers)];
        self
    }

    /// The indices of the cells `mips` and `layers` cover, whose bounds are
    /// among the texture's.
    pub(crate) fn cells(
        &self,
        mips: &Range<u32>,
        layers: &Range<u32>,
    ) -> impl Iterator<Item = usize> + use<T> {
        let intervals = |bounds: &[u32], range: &Range<u32>| {
            let at = |bound| {
                bounds
                    .binary_search(&bound)
                    .expect("the texture is split at every use's bounds")
            };
            at(range.start)..at(range.end)
        };
        let columns = self.layers.len() - 1;
        let layers = intervals(&self.layers, layers);
        intervals(&self.mips, mips)
            .flat_map(move |mip| layers.clone().map(move |layer| mip * columns + layer))
    }

    /// The mips and layers of the cell at `index`.
    pub(crate) fn bounds(&self, index: usize) -> (Range<u32>, Range<u32>) {
        let columns = self.layers.len() - 1;
        let (mip, layer) = (index / columns, index % columns);
        (
            self.mips[mip]..self.mips[mip + 1],
            self.layers[layer]..self.layers[layer + 1],
        )
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the subresource grid.

    use super::*;

    /// A texture is split at every bound of its uses, and each use covers
    /// the cells between its bounds, which give back their own bounds.
    #[test]
    fn uses_cover_the_cells_between_their_bounds() {
        let mut grid = Grid::<()>::default();
        grid.split(&(0..4), &(0..2));
        grid.split(&(1..2), &(1..2));
        let grid = grid.finish();
        // Mips split at 0, 1, 2 and 4, layers at 0, 1 and 2.
        assert_eq!(grid.cells.len(), 6);
        assert_eq!(grid.cells(&(1..2), &(0..2)).collect::<Vec<_>>(), [2, 3]);
        assert_eq!(grid.cells(&(0..4), &(1..2)).collect::<Vec<_>>(), [1, 3, 5]);
        assert_eq!(grid.bounds(5), (2..4, 1..2));
        assert_eq!(grid.bounds(0), (0..1, 0..1));
    }
}
