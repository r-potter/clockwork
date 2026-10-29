//! Samplers in the Vulkan backend (design 0002, "Binding model").
//!
//! A sampler is only a slot in the sampler heap. `VK_EXT_descriptor_heap`
//! writes its descriptor straight from a `VkSamplerCreateInfo`, so there's
//! no `VkSampler` to create or destroy. The [`VulkanDevice`]'s sampler table
//! holds each sampler's slot.
//!
//! Lock order: the sampler table is never held while a heap's slot
//! allocator is locked, so the two can't deadlock.

#[cfg(test)]
mod gpu_tests;

use std::slice;
use std::sync::{PoisonError, RwLockReadGuard, RwLockWriteGuard};

use ash::vk;

use super::VulkanDevice;
use super::error::error;
use crate::handle::HandleTable;
use crate::{
    AddressMode, DescriptorIndex, Filter, HeapKind, RhiError, RhiResult, Sampler, SamplerDesc,
};

/// A sampler in the sampler table.
#[derive(Debug)]
pub(super) struct SamplerEntry {
    /// The sampler's slot in the sampler heap.
    slot: u32,
}

impl From<Filter> for vk::Filter {
    /// The Vulkan filter for filtering texels with `filter`.
    fn from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => Self::NEAREST,
            Filter::Linear => Self::LINEAR,
        }
    }
}

impl From<Filter> for vk::SamplerMipmapMode {
    /// The Vulkan mipmap mode for filtering between mips with `filter`.
    fn from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => Self::NEAREST,
            Filter::Linear => Self::LINEAR,
        }
    }
}

impl From<AddressMode> for vk::SamplerAddressMode {
    /// The Vulkan address mode for `mode`.
    fn from(mode: AddressMode) -> Self {
        match mode {
            AddressMode::Repeat => Self::REPEAT,
            AddressMode::MirroredRepeat => Self::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => Self::CLAMP_TO_EDGE,
        }
    }
}

/// The create info `desc`'s descriptor is written from: its filters and
/// address modes, sampling every mip with no bias, anisotropy or
/// comparison.
pub(super) fn sampler_create_info(desc: &SamplerDesc<'_>) -> vk::SamplerCreateInfo<'static> {
    let [u, v, w] = desc.address_modes.map(vk::SamplerAddressMode::from);
    vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::from(desc.mag_filter))
        .min_filter(vk::Filter::from(desc.min_filter))
        .mipmap_mode(vk::SamplerMipmapMode::from(desc.mip_filter))
        .address_mode_u(u)
        .address_mode_v(v)
        .address_mode_w(w)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE)
}

impl VulkanDevice {
    /// Writes a descriptor for `desc` into a free slot in the sampler heap,
    /// and adds the sampler to the sampler table.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::DescriptorHeapFull`] if the sampler heap has no
    /// free slot, or [`RhiError::Other`] if the descriptor can't be written.
    pub(super) fn create_sampler_entry(&self, desc: &SamplerDesc<'_>) -> RhiResult<Sampler> {
        let heap = &self.heaps.sampler;
        let slot = heap
            .allocate_slot()
            .ok_or_else(|| RhiError::DescriptorHeapFull {
                heap: HeapKind::Sampler,
                label: desc.label.to_owned(),
            })?;
        let create_info = sampler_create_info(desc);
        let slot_range = heap.slot_address_range(slot);
        // SAFETY: `create_info` is valid for the call, and `slot_range` is the
        // slot's range in the heap's mapped memory, a stride long, which holds
        // a sampler descriptor. The slot was just handed to this thread, so
        // nothing else writes to the range, and no submitted work reads it,
        // since a slot is only freed once nothing can.
        let written = unsafe {
            self.device.descriptor_heap().write_sampler_descriptors(
                slice::from_ref(&create_info),
                slice::from_ref(&slot_range),
            )
        };
        if let Err(result) = written {
            heap.free_slot(slot);
            return Err(error(
                result,
                "write the descriptor for sampler",
                desc.label,
            ));
        }
        let entry = SamplerEntry { slot };
        Ok(Sampler::from_handle(
            self.lock_samplers_for_writing().insert(desc.label, entry),
        ))
    }

    /// Removes `sampler` from the table, and frees its slot.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_sampler_entry(&self, sampler: Sampler) {
        let (_, entry) = self.lock_samplers_for_writing().remove(sampler.handle());
        // No submission so far reads the descriptor once they've completed.
        // Deferred destruction, later in `rhi-commands`, replaces the wait.
        self.wait_for_submissions();
        self.free_sampler(entry);
    }

    /// Frees the slot of a sampler taken out of the table.
    pub(super) fn free_sampler(&self, entry: SamplerEntry) {
        self.heaps.sampler.free_slot(entry.slot);
    }

    /// The slot `sampler`'s descriptor is in, for shaders to index.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn sampler_entry_index(&self, sampler: Sampler) -> DescriptorIndex {
        DescriptorIndex(self.lock_samplers_for_reading().get(sampler.handle()).slot)
    }

    /// Locks the sampler table for reading, recovering it if a panic
    /// poisoned it: no table operation panics halfway through, so it's
    /// consistent.
    fn lock_samplers_for_reading(&self) -> RwLockReadGuard<'_, HandleTable<SamplerEntry>> {
        self.samplers.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the sampler table for writing, recovering it as
    /// [`lock_samplers_for_reading`](Self::lock_samplers_for_reading) does.
    fn lock_samplers_for_writing(&self) -> RwLockWriteGuard<'_, HandleTable<SamplerEntry>> {
        self.samplers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the sampler translations to Vulkan.

    use std::collections::HashSet;

    use super::*;

    /// A description with every filter `filter` and every address mode
    /// `mode`.
    fn desc(filter: Filter, mode: AddressMode) -> SamplerDesc<'static> {
        SamplerDesc {
            label: "sampler",
            mag_filter: filter,
            min_filter: filter,
            mip_filter: filter,
            address_modes: [mode; 3],
        }
    }

    /// Each filter maps to its Vulkan filter, and to the mipmap mode of the
    /// same name between mips.
    #[test]
    fn filters_map_to_vulkan_filters() {
        for (filter, vk_filter, mipmap_mode) in [
            (
                Filter::Nearest,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (
                Filter::Linear,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
            ),
        ] {
            let create_info = sampler_create_info(&desc(filter, AddressMode::Repeat));
            assert_eq!(create_info.mag_filter, vk_filter, "{filter:?}");
            assert_eq!(create_info.min_filter, vk_filter, "{filter:?}");
            assert_eq!(create_info.mipmap_mode, mipmap_mode, "{filter:?}");
        }
        let filters: HashSet<_> = Filter::ALL.into_iter().map(vk::Filter::from).collect();
        assert_eq!(filters.len(), Filter::ALL.len());
    }

    /// Each address mode maps to its Vulkan address mode.
    #[test]
    fn address_modes_map_to_vulkan_address_modes() {
        for (mode, vk_mode) in [
            (AddressMode::Repeat, vk::SamplerAddressMode::REPEAT),
            (
                AddressMode::MirroredRepeat,
                vk::SamplerAddressMode::MIRRORED_REPEAT,
            ),
            (
                AddressMode::ClampToEdge,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ),
        ] {
            let create_info = sampler_create_info(&desc(Filter::Linear, mode));
            assert_eq!(
                [
                    create_info.address_mode_u,
                    create_info.address_mode_v,
                    create_info.address_mode_w,
                ],
                [vk_mode; 3],
                "{mode:?}",
            );
        }
        let modes: HashSet<_> = AddressMode::ALL
            .into_iter()
            .map(vk::SamplerAddressMode::from)
            .collect();
        assert_eq!(modes.len(), AddressMode::ALL.len());
    }

    /// The three filters and three address modes are translated apart, so
    /// none is taken from another.
    #[test]
    fn each_filter_and_axis_is_translated_apart() {
        let create_info = sampler_create_info(&SamplerDesc {
            label: "mixed",
            mag_filter: Filter::Linear,
            min_filter: Filter::Nearest,
            mip_filter: Filter::Linear,
            address_modes: [
                AddressMode::ClampToEdge,
                AddressMode::Repeat,
                AddressMode::MirroredRepeat,
            ],
        });
        assert_eq!(create_info.mag_filter, vk::Filter::LINEAR);
        assert_eq!(create_info.min_filter, vk::Filter::NEAREST);
        assert_eq!(create_info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(
            create_info.address_mode_u,
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        );
        assert_eq!(create_info.address_mode_v, vk::SamplerAddressMode::REPEAT);
        assert_eq!(
            create_info.address_mode_w,
            vk::SamplerAddressMode::MIRRORED_REPEAT
        );
    }

    /// Every mip is sampled, with no bias, anisotropy or comparison.
    #[test]
    fn every_mip_is_sampled_without_bias() {
        let create_info = sampler_create_info(&desc(Filter::Linear, AddressMode::Repeat));
        assert_eq!(create_info.min_lod, 0.0);
        assert_eq!(create_info.max_lod, vk::LOD_CLAMP_NONE);
        assert_eq!(create_info.mip_lod_bias, 0.0);
        assert_eq!(create_info.anisotropy_enable, vk::FALSE);
        assert_eq!(create_info.compare_enable, vk::FALSE);
    }
}
