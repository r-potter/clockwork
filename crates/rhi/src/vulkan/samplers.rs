//! Samplers in the Vulkan backend (design 0002, "Binding model").
//!
//! A sampler is only a slot in the sampler heap. `VK_EXT_descriptor_heap`
//! writes its descriptor straight from a `VkSamplerCreateInfo`, so there's
//! no `VkSampler` to create or destroy. This module translates a
//! [`SamplerDesc`] into the create info its descriptor is written from.

use ash::vk;

use crate::{AddressMode, Filter, SamplerDesc};

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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Vulkan backend creates samplers in the next PR"
    )
)]
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
