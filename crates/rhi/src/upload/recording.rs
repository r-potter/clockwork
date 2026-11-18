//! What an uploader records between two flushes: the bytes it stages, the
//! copies that read them, and the barriers around the copies (design 0002,
//! "Streaming" and "Texture copies").
//!
//! Copies go to the transfer queue, unless it can't copy a texture region,
//! when they go to the graphics queue, which can copy any. Each queue's
//! copies become one list at the flush, in three steps:
//!
//! 1. a barrier for each texture subresource copied to, from
//!    [`Access::None`] to [`Access::CopyWrite`], discarding its contents;
//! 2. the copies, in the order they were uploaded;
//! 3. a barrier for each range of a buffer and each subresource, from
//!    [`Access::CopyWrite`] to the accesses it's uploaded for, which on the
//!    transfer queue also releases it to the graphics queue when they're
//!    different queues.
//!
//! A buffer's barriers cover only the bytes uploaded, so the rest of the
//! buffer keeps its owner and contents. A subresource's cover all of it,
//! once a flush, and discard what it held, so a flush fills each mip it
//! uploads to whole, in one region or several. Copies in one list aren't
//! ordered, so uploads in one flush can't overlap: two that write the same
//! bytes, or the same subresource for different accesses or on different
//! queues, panic.
//!
//! Each upload is checked against what's uploaded to the same buffer or
//! subresource, found by a map, and the flush checks that regions and
//! ranges don't overlap by sorting them, so a flush of many uploads stays
//! cheap.
//!
//! The releases are also the acquire barriers rendering records before it
//! first uses what was uploaded. The graphics queue's copies need none.
//!
//! Recording is ordinary code over [`GpuResources`], tested with a fake of
//! it. Translating and submitting the lists is the uploader's flush.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::staging::Staging;
use crate::copy::mip_extent;
use crate::{
    Access, AccessSet, Barrier, Buffer, CommandList, Extent3d, GpuAddress, GpuResources, QueueId,
    RhiResult, Subresource, Texture, TextureCopy, TextureDataLayout, TextureUsage,
    TransferGranularity,
};

/// What the uploader needs to know about its transfer queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransferQueue {
    /// Which queue it is.
    pub(crate) id: QueueId,
    /// The texture regions it can copy.
    pub(crate) granularity: TransferGranularity,
}

/// A copy recorded for a flush.
#[derive(Debug)]
enum StagedCopy {
    /// From staged bytes to a buffer.
    Buffer {
        /// The staged bytes.
        source: GpuAddress,
        /// Where they're copied to.
        destination: GpuAddress,
        /// How many bytes are copied.
        size: u64,
    },
    /// From tightly packed staged bytes to a texture region.
    Texture {
        /// The staged bytes.
        source: GpuAddress,
        /// How they're laid out.
        layout: TextureDataLayout,
        /// The region they're copied to.
        region: TextureCopy,
    },
}

/// A buffer uploaded to in a flush.
#[derive(Debug)]
struct BufferUpload {
    /// The buffer.
    buffer: Buffer,
    /// How it's used after the upload.
    then: AccessSet,
    /// The byte ranges copied to, by offset: in the order they were
    /// uploaded, until the finish sorts them and joins those that touch.
    ranges: Vec<Range<u64>>,
}

/// A texture subresource, one mip in some layers, uploaded to in a flush.
#[derive(Debug)]
struct TextureUpload {
    /// The texture.
    texture: Texture,
    /// The mip.
    mip: u32,
    /// The layers.
    layers: Range<u32>,
    /// The mip's extent, in texels, which the regions fill.
    extent: Extent3d,
    /// How it's used after the upload.
    then: AccessSet,
    /// The regions copied to, as each one's origin and extent.
    regions: Vec<([u32; 3], Extent3d)>,
}

impl TextureUpload {
    /// The subresource it uploads to.
    fn subresource(&self) -> Subresource {
        Subresource {
            mips: self.mip..self.mip + 1,
            layers: self.layers.clone(),
        }
    }
}

/// Which queue's copies an upload is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnQueue {
    /// The transfer queue's.
    Transfer,
    /// The graphics queue's.
    Graphics,
}

/// The copies for one queue in a flush, and what they upload to.
#[derive(Debug, Default)]
struct QueueCopies {
    /// The copies, in the order they were uploaded.
    copies: Vec<StagedCopy>,
    /// The buffers they copy to.
    buffers: Vec<BufferUpload>,
    /// The texture subresources they copy to.
    textures: Vec<TextureUpload>,
}

/// The lists a flush translates, and the acquire barriers rendering needs.
#[derive(Debug)]
pub(crate) struct Recorded {
    /// The list for the transfer queue, if anything was copied on it.
    pub(crate) transfer: Option<CommandList>,
    /// The list for the graphics queue, if a texture region the transfer
    /// queue can't copy was uploaded.
    pub(crate) graphics: Option<CommandList>,
    /// The barriers rendering records on the graphics queue before it first
    /// uses what was uploaded: the transfer queue's releases, when it isn't
    /// the graphics queue.
    pub(crate) acquire: Vec<Barrier>,
}

/// An uploader's staging and the copies it has recorded since the last
/// flush.
#[derive(Debug)]
pub(crate) struct Recording {
    /// The resources uploaded to.
    resources: Arc<dyn GpuResources>,
    /// The uploader's label, which names its lists.
    label: String,
    /// Where the bytes are staged.
    staging: Staging,
    /// The transfer queue.
    transfer: TransferQueue,
    /// The graphics queue, which copies what the transfer queue can't, and
    /// acquires what it uploads.
    graphics: QueueId,
    /// The transfer queue's copies.
    on_transfer: QueueCopies,
    /// The graphics queue's copies.
    on_graphics: QueueCopies,
    /// Where each buffer uploaded to is in the transfer queue's buffers.
    buffer_uploads: HashMap<Buffer, usize>,
    /// Where the uploads to each mip of a texture are, by queue and index in
    /// that queue's textures: one for each set of layers.
    texture_uploads: HashMap<(Texture, u32), Vec<(OnQueue, usize)>>,
}

impl Recording {
    /// Recording for the uploader labelled `label`, copying on `transfer`,
    /// and on `graphics` what `transfer` can't.
    pub(crate) fn new(
        resources: Arc<dyn GpuResources>,
        transfer: TransferQueue,
        graphics: QueueId,
        label: &str,
    ) -> Self {
        Self {
            staging: Staging::new(resources.clone(), label),
            resources,
            label: label.to_owned(),
            transfer,
            graphics,
            on_transfer: QueueCopies::default(),
            on_graphics: QueueCopies::default(),
            buffer_uploads: HashMap::new(),
            texture_uploads: HashMap::new(),
        }
    }

    /// The staging, for the flush to end its batch.
    pub(crate) fn staging(&mut self) -> &mut Staging {
        &mut self.staging
    }

    /// Stages `data` and records its copy to `buffer` at `offset`, on the
    /// transfer queue, leaving those bytes of the buffer ready for `then`.
    ///
    /// # Errors
    ///
    /// Returns an error if a staging buffer can't be created, having
    /// recorded nothing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, `data` is empty or doesn't fit in the
    /// buffer at `offset`, `then` isn't an access set a buffer can have, or
    /// the buffer was uploaded to in this flush for other accesses. Uploads
    /// to overlapping bytes panic at [`finish`](Self::finish).
    pub(crate) fn upload_buffer(
        &mut self,
        buffer: Buffer,
        offset: u64,
        data: &[u8],
        then: AccessSet,
    ) -> RhiResult<()> {
        let label = &self.label;
        assert!(
            !data.is_empty(),
            "uploader \"{label}\" uploads no bytes to a buffer"
        );
        Barrier::buffer(buffer, Access::CopyWrite, then).check(label);
        let size = data.len() as u64;
        let buffer_size = self.resources.buffer_size(buffer);
        let range = offset
            .checked_add(size)
            .filter(|&end| end <= buffer_size)
            .map(|end| offset..end)
            .unwrap_or_else(|| {
                panic!(
                    "uploader \"{label}\" uploads {size} bytes at offset {offset} of a buffer of \
                     {buffer_size} bytes"
                )
            });
        // The buffer is in the address space, so an offset inside it is too.
        let destination = GpuAddress(self.resources.buffer_address(buffer).0 + offset);
        let existing = self.buffer_uploads.get(&buffer).copied();
        if let Some(index) = existing {
            let upload = &self.on_transfer.buffers[index];
            assert!(
                upload.then == then,
                "uploader \"{label}\" uploads to a buffer for {then:?}, which it uploaded to for \
                 {:?} in this flush",
                upload.then,
            );
        }

        let source = self.staging.write(data, 1)?;
        let uploads = &mut self.on_transfer.buffers;
        match existing {
            Some(index) => uploads[index].ranges.push(range),
            None => {
                self.buffer_uploads.insert(buffer, uploads.len());
                uploads.push(BufferUpload {
                    buffer,
                    then,
                    ranges: vec![range],
                });
            }
        }
        self.on_transfer.copies.push(StagedCopy::Buffer {
            source,
            destination,
            size,
        });
        Ok(())
    }

    /// Stages `data`, tightly packed texel blocks, and records its copy to
    /// the texture region `region` names, leaving the region's mip and
    /// layers ready for `then`. The copy is on the transfer queue if it can
    /// copy the region, and on the graphics queue otherwise.
    ///
    /// The barrier before the copy discards what the mip and layers held
    /// before this flush, so the flush must fill them whole, in one region
    /// or several.
    ///
    /// # Errors
    ///
    /// Returns an error if a staging buffer can't be created, having
    /// recorded nothing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the texture can't be copied to, the
    /// region isn't in its mip, `data` isn't as long as the region's tightly
    /// packed blocks, `then` isn't an access set a texture can have, or the
    /// region's layers overlap others uploaded to in its mip in this flush
    /// without being the same, or are the same but for other accesses or on
    /// the other queue. Overlapping regions, and mips not filled whole,
    /// panic at [`finish`](Self::finish), and the rest of the copy is
    /// checked when it's translated.
    pub(crate) fn upload_texture(
        &mut self,
        region: TextureCopy,
        data: &[u8],
        then: AccessSet,
    ) -> RhiResult<()> {
        let label = &self.label;
        let info = self.resources.texture_info(region.texture);
        assert!(
            info.usage.contains(TextureUsage::COPY_DESTINATION),
            "uploader \"{label}\" uploads to a texture that can't be copied to"
        );
        let extent = mip_extent(info.extent, region.mip);
        let ends = [extent.width, extent.height, extent.depth];
        let sizes = [
            region.extent.width,
            region.extent.height,
            region.extent.depth,
        ];
        assert!(
            (0..3).all(|axis| {
                region.origin[axis]
                    .checked_add(sizes[axis])
                    .is_some_and(|end| end <= ends[axis])
            }),
            "uploader \"{label}\" uploads to a region at {:?} of {:?}, which isn't in mip {} of \
             {extent:?}",
            region.origin,
            region.extent,
            region.mip,
        );
        let layout = TextureDataLayout::tightly_packed(info.format, region.extent);
        let layers = u64::from(region.layers.end.saturating_sub(region.layers.start));
        let expected = [
            u64::from(layout.rows_per_image),
            u64::from(region.extent.depth),
            layers,
        ]
        .into_iter()
        .try_fold(u64::from(layout.bytes_per_row), u64::checked_mul);
        assert!(
            expected.is_some_and(|expected| expected > 0 && data.len() as u64 == expected),
            "uploader \"{label}\" uploads {} bytes to a region of {:?} in {:?} layers, which \
             needs {} bytes of tightly packed {:?}",
            data.len(),
            region.extent,
            region.layers,
            expected.map_or_else(|| "more than 2^64".to_owned(), |bytes| bytes.to_string()),
            info.format,
        );
        let upload = TextureUpload {
            texture: region.texture,
            mip: region.mip,
            layers: region.layers.clone(),
            extent,
            then,
            regions: vec![(region.origin, region.extent)],
        };
        Barrier::subresource(
            region.texture,
            upload.subresource(),
            Access::CopyWrite,
            then,
        )
        .check(label);

        let on = if self
            .transfer
            .granularity
            .allows(info.format, info.extent, &region)
        {
            OnQueue::Transfer
        } else {
            OnQueue::Graphics
        };
        let key = (region.texture, region.mip);
        let mut existing = None;
        for &(queue, index) in self.texture_uploads.get(&key).into_iter().flatten() {
            let other = &self.copies(queue).textures[index];
            if other.layers.start >= upload.layers.end || upload.layers.start >= other.layers.end {
                continue;
            }
            assert!(
                other.layers == upload.layers,
                "uploader \"{label}\" uploads to mip {} of layers {:?} of a texture, which overlap \
                 layers {:?} it uploaded to in this flush",
                region.mip,
                region.layers,
                other.layers,
            );
            assert!(
                other.then == then,
                "uploader \"{label}\" uploads to mip {} of layers {:?} of a texture for {then:?}, \
                 having uploaded to it for {:?} in this flush",
                region.mip,
                region.layers,
                other.then,
            );
            assert!(
                queue == on,
                "uploader \"{label}\" uploads to mip {} of layers {:?} of a texture on one queue, \
                 having uploaded to it on another in this flush",
                region.mip,
                region.layers,
            );
            existing = Some(index);
        }

        let alignment = self
            .resources
            .capabilities()
            .limits
            .copy_offset_alignment(info.format);
        let source = self.staging.write(data, alignment)?;
        let copies = match on {
            OnQueue::Transfer => &mut self.on_transfer,
            OnQueue::Graphics => &mut self.on_graphics,
        };
        match existing {
            Some(index) => copies.textures[index]
                .regions
                .push((region.origin, region.extent)),
            None => {
                self.texture_uploads
                    .entry(key)
                    .or_default()
                    .push((on, copies.textures.len()));
                copies.textures.push(upload);
            }
        }
        copies.copies.push(StagedCopy::Texture {
            source,
            layout,
            region,
        });
        Ok(())
    }

    /// The copies on the queue `on`.
    fn copies(&self, on: OnQueue) -> &QueueCopies {
        match on {
            OnQueue::Transfer => &self.on_transfer,
            OnQueue::Graphics => &self.on_graphics,
        }
    }

    /// The lists for the copies recorded since the last flush, and the
    /// acquire barriers rendering needs, leaving nothing recorded. The
    /// staged bytes stay in the staging's batch until the flush ends it.
    ///
    /// # Panics
    ///
    /// Panics if two uploads to a buffer overlap, two regions of a mip
    /// overlap, or a mip uploaded to isn't filled whole.
    pub(crate) fn finish(&mut self) -> Recorded {
        let mut on_transfer = std::mem::take(&mut self.on_transfer);
        let on_graphics = std::mem::take(&mut self.on_graphics);
        self.buffer_uploads.clear();
        self.texture_uploads.clear();
        for upload in &mut on_transfer.buffers {
            upload.ranges = join_ranges(&self.label, std::mem::take(&mut upload.ranges));
        }
        for upload in on_transfer.textures.iter().chain(&on_graphics.textures) {
            check_regions(&self.label, upload);
        }

        let release =
            (self.transfer.id != self.graphics).then_some((self.transfer.id, self.graphics));
        let mut acquire = Vec::new();
        let transfer = list(&self.label, on_transfer, release, &mut acquire);
        let graphics = list(
            &format!("{} (graphics)", self.label),
            on_graphics,
            None,
            &mut acquire,
        );
        Recorded {
            transfer,
            graphics,
            acquire,
        }
    }
}

/// Checks that no two of a buffer upload's ranges overlap, by sorting them
/// and comparing neighbours, and returns them sorted, with ranges that
/// touch joined, so that adjacent uploads share a barrier.
///
/// # Panics
///
/// Panics, naming the uploader labelled `label`, if two overlap.
fn join_ranges(label: &str, mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut joined: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match joined.last_mut() {
            Some(last) => {
                assert!(
                    last.end <= range.start,
                    "uploader \"{label}\" uploads to bytes {range:?} of a buffer, which overlap \
                     bytes it uploaded to in this flush",
                );
                if last.end == range.start {
                    last.end = range.end;
                } else {
                    joined.push(range);
                }
            }
            None => joined.push(range),
        }
    }
    joined
}

/// Checks that no two of a subresource upload's regions overlap, and that
/// together they fill its mip, since the barrier before them discards what
/// it held.
///
/// Regions are swept in order of where they start along one axis, keeping
/// those whose range along it reaches the current one's start, so each is
/// compared only with those beside it along that axis, rather than with
/// every other. The axis is the one along which the regions start in the
/// most places, so that rows of a mip are swept down it, columns across
/// it, and a grid of tiles either way.
///
/// # Panics
///
/// Panics, naming the uploader labelled `label`, if two regions overlap, or
/// they don't fill the mip.
fn check_regions(label: &str, upload: &TextureUpload) {
    let mut regions = upload.regions.clone();
    let starts = |axis: usize| {
        let mut starts: Vec<_> = regions.iter().map(|(origin, _)| origin[axis]).collect();
        starts.sort_unstable();
        starts.dedup();
        starts.len()
    };
    let axis = (0..3)
        .max_by_key(|&axis| starts(axis))
        .expect("there are three axes");
    regions.sort_unstable_by_key(|&(origin, _)| origin[axis]);
    let end = |&(origin, extent): &([u32; 3], Extent3d)| {
        u64::from(origin[axis]) + u64::from([extent.width, extent.height, extent.depth][axis])
    };
    let mut active: Vec<([u32; 3], Extent3d)> = Vec::new();
    let mut filled = 0u64;
    for region in regions {
        let (origin, extent) = region;
        active.retain(|other| end(other) > u64::from(origin[axis]));
        if let Some(&(other_origin, other_extent)) =
            active.iter().find(|&&(other_origin, other_extent)| {
                boxes_overlap(origin, extent, other_origin, other_extent)
            })
        {
            panic!(
                "uploader \"{label}\" uploads to regions at {other_origin:?} of {other_extent:?} \
                 and at {origin:?} of {extent:?} of mip {} of layers {:?} of a texture in one \
                 flush, which overlap",
                upload.mip, upload.layers,
            );
        }
        filled += texels(extent);
        active.push(region);
    }
    // The regions are in the mip and don't overlap, so they fill it if their
    // texels add up to its.
    assert!(
        filled == texels(upload.extent),
        "uploader \"{label}\" uploads {filled} of the {} texels of mip {} of layers {:?} of a \
         texture in one flush, but a flush discards what a mip held, so it fills the mip whole",
        texels(upload.extent),
        upload.mip,
        upload.layers,
    );
}

/// How many texels a region of `extent` has.
fn texels(extent: Extent3d) -> u64 {
    u64::from(extent.width) * u64::from(extent.height) * u64::from(extent.depth)
}

/// The list labelled `label` for one queue's copies, with the barriers
/// around them, or `None` if there are none. The barriers after them move
/// what they upload from the queue `release` names first to the second,
/// and are added to `acquire`, if `release` names queues.
fn list(
    label: &str,
    copies: QueueCopies,
    release: Option<(QueueId, QueueId)>,
    acquire: &mut Vec<Barrier>,
) -> Option<CommandList> {
    if copies.copies.is_empty() {
        return None;
    }
    let mut list = CommandList::new(label);
    let before: Vec<_> = copies
        .textures
        .iter()
        .map(|upload| {
            Barrier::subresource(
                upload.texture,
                upload.subresource(),
                Access::None,
                Access::CopyWrite,
            )
        })
        .collect();
    list.barrier(&before);
    for copy in copies.copies {
        match copy {
            StagedCopy::Buffer {
                source,
                destination,
                size,
            } => list.copy_buffer(source, destination, size),
            StagedCopy::Texture {
                source,
                layout,
                region,
            } => list.copy_buffer_to_texture(source, layout, region),
        }
    }
    let after: Vec<_> = copies
        .buffers
        .iter()
        .flat_map(|upload| {
            upload.ranges.iter().map(|range| {
                Barrier::buffer_range(upload.buffer, range.clone(), Access::CopyWrite, upload.then)
            })
        })
        .chain(copies.textures.iter().map(|upload| {
            Barrier::subresource(
                upload.texture,
                upload.subresource(),
                Access::CopyWrite,
                upload.then,
            )
        }))
        .map(|barrier| match release {
            Some((from, to)) => barrier.with_queue_transfer(from, to),
            None => barrier,
        })
        .collect();
    list.barrier(&after);
    if release.is_some() {
        acquire.extend(after);
    }
    Some(list)
}

/// Whether the regions at `a` of `a_extent` and at `b` of `b_extent` share
/// any texel.
fn boxes_overlap(a: [u32; 3], a_extent: Extent3d, b: [u32; 3], b_extent: Extent3d) -> bool {
    let a_ends = [a_extent.width, a_extent.height, a_extent.depth];
    let b_ends = [b_extent.width, b_extent.height, b_extent.depth];
    (0..3).all(|axis| {
        let (a_start, b_start) = (u64::from(a[axis]), u64::from(b[axis]));
        a_start < b_start + u64::from(b_ends[axis]) && b_start < a_start + u64::from(a_ends[axis])
    })
}

#[cfg(test)]
mod tests;
