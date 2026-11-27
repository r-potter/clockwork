//! The textures and buffers the executor creates for a plan (design 0003,
//! "Executing", R6): the transient and persistent resources its passes use,
//! with usage derived from their accesses.
//!
//! A resource is created only if a pass that runs uses it, so a resource
//! only culled passes use, which would have no usage, isn't. The output and
//! imports are the caller's, so they're never created.
//!
//! What's created is kept as plain data, compared from one compile to the
//! next, so a persistent resource whose description is unchanged keeps its
//! contents. Its usage is part of its description: a resource's usage is
//! fixed when it's created, so a new plan that uses a persistent resource
//! in a new way, such as sampling a texture it only wrote, recreates it,
//! and its contents are lost.

use clockwork_rhi::{
    Access, AccessSet, BufferDesc, BufferUsage, MemoryLocation, TextureDesc, TextureUsage,
};

use crate::desc::{BufferKind, TextureKind};
use crate::validate::TextureShape;
use crate::{GraphDesc, Plan, ResourceUse};

/// The resources the executor creates for a plan, each by its index in the
/// description, or `None` for one it doesn't create.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Created {
    /// Each texture's, indexed by `TextureId`.
    pub(crate) textures: Vec<Option<CreatedTexture>>,
    /// Each buffer's, indexed by `BufferId`.
    pub(crate) buffers: Vec<Option<CreatedBuffer>>,
}

/// A texture the executor creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CreatedTexture {
    /// Whether its contents are kept from one frame to the next.
    pub(crate) persistent: bool,
    /// Its shape.
    pub(crate) shape: TextureShape,
    /// How its uses use it.
    pub(crate) usage: TextureUsage,
}

impl CreatedTexture {
    /// Its description for the RHI, named `label`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn desc(self, label: &str) -> TextureDesc<'_> {
        TextureDesc {
            label,
            dimension: self.shape.dimension,
            format: self.shape.format,
            extent: self.shape.extent,
            mip_levels: self.shape.mips,
            array_layers: self.shape.layers,
            usage: self.usage,
        }
    }
}

/// A buffer the executor creates, in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CreatedBuffer {
    /// Whether its contents are kept from one frame to the next.
    pub(crate) persistent: bool,
    /// Its size, in bytes.
    pub(crate) size: u64,
    /// How its uses use it, beyond copies, which every buffer allows.
    pub(crate) usage: BufferUsage,
}

impl CreatedBuffer {
    /// Its description for the RHI, named `label`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn desc(self, label: &str) -> BufferDesc<'_> {
        BufferDesc {
            label,
            size: self.size,
            usage: self.usage,
            location: MemoryLocation::Device,
        }
    }
}

/// The resources the executor creates for `plan`, compiled from `desc`:
/// each transient or persistent texture and buffer a pass that runs uses,
/// with the usage every access of those passes needs.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
)]
pub(crate) fn created(desc: &GraphDesc, plan: &Plan) -> Created {
    let mut texture_usage = vec![None::<TextureUsage>; desc.textures.len()];
    let mut buffer_usage = vec![None::<BufferUsage>; desc.buffers.len()];
    for planned in &plan.passes {
        for used in &plan.expanded[planned.pass].pass.uses {
            match &used.resource {
                ResourceUse::Texture(part) => {
                    let usage = &mut texture_usage[part.texture.index()];
                    *usage =
                        Some(usage.unwrap_or(TextureUsage::NONE) | texture_usage_of(used.access));
                }
                ResourceUse::Buffer(buffer) => {
                    let usage = &mut buffer_usage[buffer.index()];
                    *usage =
                        Some(usage.unwrap_or(BufferUsage::COPIES) | buffer_usage_of(used.access));
                }
                ResourceUse::Repeated(_) => {
                    unreachable!("expanding a valid description resolves every relative mip")
                }
            }
        }
    }
    let textures = desc
        .textures
        .iter()
        .zip(&plan.textures)
        .zip(texture_usage)
        .map(|((texture, shape), usage)| {
            let persistent = match texture.kind {
                TextureKind::Transient(_) => false,
                TextureKind::Persistent(_) => true,
                TextureKind::Output | TextureKind::Imported(_) => return None,
            };
            let shape = shape.expect("a texture the graph creates has a shape");
            usage.map(|usage| CreatedTexture {
                persistent,
                shape,
                usage,
            })
        })
        .collect();
    let buffers = desc
        .buffers
        .iter()
        .zip(buffer_usage)
        .map(|(buffer, usage)| {
            let (persistent, size) = match buffer.kind {
                BufferKind::Transient { size } => (false, size),
                BufferKind::Persistent { size } => (true, size),
                BufferKind::Imported(_) => return None,
            };
            usage.map(|usage| CreatedBuffer {
                persistent,
                size,
                usage,
            })
        })
        .collect();
    Created { textures, buffers }
}

/// The texture usage the accesses in `set` need.
fn texture_usage_of(set: AccessSet) -> TextureUsage {
    set.iter().fold(TextureUsage::NONE, |usage, access| {
        usage
            | match access {
                Access::VertexShaderRead
                | Access::FragmentShaderRead
                | Access::ComputeShaderRead
                | Access::ComputeShaderWrite => TextureUsage::STORAGE,
                Access::FragmentShaderSampledRead | Access::ComputeShaderSampledRead => {
                    TextureUsage::SAMPLED
                }
                Access::ColorAttachmentRead | Access::ColorAttachmentWrite => {
                    TextureUsage::COLOUR_ATTACHMENT
                }
                Access::DepthAttachmentRead | Access::DepthAttachmentWrite => {
                    TextureUsage::DEPTH_ATTACHMENT
                }
                Access::CopyRead => TextureUsage::COPY_SOURCE,
                Access::CopyWrite => TextureUsage::COPY_DESTINATION,
                // Validation allows none of these of a texture in a pass.
                Access::None
                | Access::IndirectRead
                | Access::IndexRead
                | Access::HostWrite
                | Access::HostRead
                | Access::Present
                | Access::Acquired => TextureUsage::NONE,
            }
    })
}

/// The buffer usage, beyond copies, the accesses in `set` need.
fn buffer_usage_of(set: AccessSet) -> BufferUsage {
    set.iter().fold(BufferUsage::COPIES, |usage, access| {
        usage
            | match access {
                Access::VertexShaderRead
                | Access::FragmentShaderRead
                | Access::ComputeShaderRead
                | Access::ComputeShaderWrite => BufferUsage::STORAGE,
                Access::IndexRead => BufferUsage::INDEX,
                Access::IndirectRead => BufferUsage::INDIRECT,
                // Copies need nothing more, and validation allows none of
                // the rest of a buffer in a pass.
                Access::CopyRead
                | Access::CopyWrite
                | Access::None
                | Access::FragmentShaderSampledRead
                | Access::ComputeShaderSampledRead
                | Access::ColorAttachmentRead
                | Access::ColorAttachmentWrite
                | Access::DepthAttachmentRead
                | Access::DepthAttachmentWrite
                | Access::HostWrite
                | Access::HostRead
                | Access::Present
                | Access::Acquired => BufferUsage::COPIES,
            }
    })
}

#[cfg(test)]
mod tests {
    //! T0 tests of the resources a plan creates, and their usage.

    use clockwork_rhi::{Extent3d, Format, TextureDimension};

    use super::*;
    use crate::validate::test_support::{finished, inputs};
    use crate::{
        Attachment, ComputeRecorder, ImportDesc, MipCount, TextureSize, TransientTexture, compile,
    };

    /// A 64 × 64 texture of `format`, with `mips` mips.
    fn texture(format: Format, mips: u32) -> TransientTexture {
        TransientTexture {
            mips: MipCount::Count(mips),
            ..TransientTexture::new(format, TextureSize::Fixed(Extent3d::new_2d(64, 64)))
        }
    }

    /// A graph that uses textures and buffers every way: rendered to and
    /// sampled, written as storage and copied, tested as depth, imported,
    /// and used only by a culled pass.
    fn graph() -> GraphDesc {
        let mut graph = GraphDesc::new("resources");
        let scene = graph.transient_texture("scene", texture(Format::Rgba16Float, 1));
        let history = graph.persistent_texture("history", texture(Format::Rgba16Float, 2));
        let depth = graph.transient_texture("depth", texture(Format::Depth32Float, 1));
        let unused = graph.transient_texture("unused", texture(Format::Rgba8Unorm, 1));
        let font = graph.import_texture(
            "font",
            ImportDesc {
                before: Access::FragmentShaderSampledRead.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        let arguments = graph.transient_buffer("arguments", 64);
        let indices = graph.persistent_buffer("indices", 128);
        let staging = graph.transient_buffer("staging", 256);
        let instances = graph.import_buffer(
            "instances",
            ImportDesc {
                before: Access::HostWrite.into(),
                after: Access::HostWrite.into(),
            },
        );
        graph
            .compute_pass("cull", ComputeRecorder::recorder("cull"))
            .write(arguments, Access::ComputeShaderWrite)
            .read(instances, Access::ComputeShaderRead);
        graph
            .raster_pass("draw", "draw")
            .read(arguments, Access::IndirectRead)
            .read(indices, Access::IndexRead)
            .read(font, Access::FragmentShaderSampledRead)
            .colour(scene, Attachment::Clear([0.0; 4]))
            .depth(depth, Attachment::Clear(0.0));
        graph
            .compute_pass("accumulate", ComputeRecorder::recorder("accumulate"))
            .read(scene, Access::ComputeShaderSampledRead)
            .write(
                history.mip(0),
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            );
        graph
            .transfer_pass("save", "save")
            .read(history.mip(0), Access::CopyRead)
            .write(staging, Access::CopyWrite)
            .side_effects();
        graph
            .transfer_pass("culled", "culled")
            .write(unused, Access::CopyWrite);
        finished(&graph)
    }

    /// Each texture a running pass uses is created with the usage of every
    /// access, and its shape, and the output, imports and a texture only a
    /// culled pass uses aren't.
    #[test]
    fn textures_get_the_usage_of_their_accesses() {
        let desc = graph();
        let plan = compile(&desc, &inputs()).expect("the graph is valid");
        let created = created(&desc, &plan);
        let usage = |index: usize| created.textures[index].map(|texture| texture.usage);
        // The output, then scene, history, depth, unused and font.
        assert_eq!(usage(0), None);
        assert_eq!(
            usage(1),
            Some(TextureUsage::COLOUR_ATTACHMENT | TextureUsage::SAMPLED)
        );
        assert_eq!(
            usage(2),
            Some(TextureUsage::STORAGE | TextureUsage::COPY_SOURCE)
        );
        assert_eq!(usage(3), Some(TextureUsage::DEPTH_ATTACHMENT));
        assert_eq!(usage(4), None);
        assert_eq!(usage(5), None);

        let history = created.textures[2].expect("history is created");
        assert!(history.persistent);
        assert!(!created.textures[1].expect("scene is created").persistent);
        assert_eq!(
            history.desc("history"),
            TextureDesc {
                label: "history",
                dimension: TextureDimension::D2,
                format: Format::Rgba16Float,
                extent: Extent3d::new_2d(64, 64),
                mip_levels: 2,
                array_layers: 1,
                usage: TextureUsage::STORAGE | TextureUsage::COPY_SOURCE,
            }
        );
    }

    /// Each buffer a running pass uses is created in device memory with the
    /// usage of every access beyond copies, and imports aren't.
    #[test]
    fn buffers_get_the_usage_of_their_accesses() {
        let desc = graph();
        let plan = compile(&desc, &inputs()).expect("the graph is valid");
        let created = created(&desc, &plan);
        // arguments, indices, staging and instances.
        assert_eq!(
            created.buffers[0],
            Some(CreatedBuffer {
                persistent: false,
                size: 64,
                usage: BufferUsage::STORAGE | BufferUsage::INDIRECT,
            })
        );
        assert_eq!(
            created.buffers[1],
            Some(CreatedBuffer {
                persistent: true,
                size: 128,
                usage: BufferUsage::INDEX,
            })
        );
        assert_eq!(
            created.buffers[2].map(|buffer| buffer.usage),
            Some(BufferUsage::COPIES)
        );
        assert_eq!(created.buffers[3], None);
        assert_eq!(
            created.buffers[1]
                .expect("indices are created")
                .desc("indices"),
            BufferDesc {
                label: "indices",
                size: 128,
                usage: BufferUsage::INDEX,
                location: MemoryLocation::Device,
            }
        );
    }
}
