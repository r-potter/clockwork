//! T0 property tests of compiling random graphs (design 0003, "Test plan"),
//! with the plan checker as their oracle (R3, R18).
//!
//! A graph is generated as a recipe, plain numbers that `proptest` can
//! shrink, which [`build`] turns into a description that's valid by
//! construction: each use is fitted to its pass's kind, and a read of a
//! transient resource's contents that no earlier pass wrote becomes a write,
//! or is left out. So every recipe compiles, and a failure to compile is a
//! bug in the generator or the compiler, never a rejected input.
//!
//! The properties: the compiler's plans have no hazards, under the schedule
//! and in any valid order; the schedule is valid and the same on every
//! run; and every barrier of a plan is needed, since the checker finds a
//! hazard in the plan without it.

use clockwork_rhi::{Access, AccessSet, Extent3d, Format, QueueKind, Subresource};
use proptest::prelude::*;

use crate::barriers::{BarrierTarget, PlannedBarrier};
use crate::check::covers_anything;
use crate::compile::compile_in_order;
use crate::dependencies::Dependencies;
use crate::expand::ExpandedPass;
use crate::schedule::{Scheduled, layout_clash, schedule};
use crate::validate::test_support::{finished, inputs};
use crate::{
    Attachment, BufferId, ComputeRecorder, GraphDesc, ImportDesc, MipCount, MipDirection, Plan,
    Schedule, TextureId, TextureSize, TextureUse, TransientTexture, check, compile,
};

/// The accesses an imported texture can be in before and after a frame.
const TEXTURE_IMPORTS: [Access; 5] = [
    Access::CopyRead,
    Access::CopyWrite,
    Access::FragmentShaderSampledRead,
    Access::ComputeShaderSampledRead,
    Access::ComputeShaderWrite,
];

/// The accesses an imported buffer can be in before and after a frame.
const BUFFER_IMPORTS: [Access; 6] = [
    Access::HostRead,
    Access::HostWrite,
    Access::CopyRead,
    Access::CopyWrite,
    Access::ComputeShaderRead,
    Access::ComputeShaderWrite,
];

/// The mips and layers a use of an imported texture names, whose shape the
/// graph doesn't know.
const IMPORT_EXTENT: u32 = 2;

/// Where a random resource comes from.
#[derive(Debug, Clone, Copy)]
enum Origin {
    /// Created by the graph, with contents that live within a frame.
    Transient,
    /// Created by the graph, with contents kept from one frame to the next.
    Persistent,
    /// The caller's, in the accesses at these indices of the resource's
    /// imports before and after the frame.
    Imported {
        /// The index of the accesses before the frame.
        before: usize,
        /// The index of the accesses after the frame.
        after: usize,
    },
}

/// A random texture.
#[derive(Debug, Clone)]
struct TextureRecipe {
    /// Where it comes from.
    origin: Origin,
    /// Whether it's a depth texture, if the graph creates it.
    depth: bool,
    /// Its mips, if the graph creates it.
    mips: u32,
    /// Its layers, if the graph creates it.
    layers: u32,
}

/// A random kind of pass.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// A transfer pass, which copies.
    Transfer,
    /// A compute pass a recorder records.
    Compute,
    /// A raster pass, which renders to the textures its uses write.
    Raster,
    /// A compute pass the graph dispatches, repeated for each mip of the
    /// first texture its uses name that has more than one.
    Chain,
}

/// A random use, fitted to its pass when the graph is built.
#[derive(Debug, Clone)]
struct UseRecipe {
    /// The resource: a texture, or a buffer past the textures, modulo how
    /// many there are.
    resource: usize,
    /// The mip, modulo the texture's mips.
    mip: u32,
    /// The layer, modulo the texture's layers.
    layer: u32,
    /// How much of the texture: 0 for all of it, 1 for one mip in every
    /// layer, and 2 for one mip of one layer.
    part: u8,
    /// Whether it writes, where the pass's kind can.
    write: bool,
    /// Which of the accesses that fit it uses, modulo how many there are.
    variant: u8,
}

/// A random pass.
#[derive(Debug, Clone)]
struct PassRecipe {
    /// Its kind.
    kind: Kind,
    /// Its uses. Those of a resource an earlier use names are left out.
    uses: Vec<UseRecipe>,
    /// Whether it's kept even if nothing reads what it writes.
    side_effects: bool,
    /// Which queue it asks for, modulo those its kind can.
    queue: u8,
}

/// A random graph.
#[derive(Debug, Clone)]
struct GraphRecipe {
    /// Its textures, besides the output.
    textures: Vec<TextureRecipe>,
    /// Its buffers, each where it comes from.
    buffers: Vec<Origin>,
    /// Its passes, before the one that writes the output.
    passes: Vec<PassRecipe>,
}

/// Where a resource comes from, with `imports` accesses to choose from
/// before and after the frame if it's imported.
fn origin(imports: usize) -> impl Strategy<Value = Origin> {
    prop_oneof![
        Just(Origin::Transient),
        Just(Origin::Persistent),
        (0..imports, 0..imports).prop_map(|(before, after)| Origin::Imported { before, after }),
    ]
}

/// Up to 4 textures, 3 buffers and 10 passes, each with up to 4 uses.
fn graphs() -> impl Strategy<Value = GraphRecipe> {
    let texture = (
        origin(TEXTURE_IMPORTS.len()),
        any::<bool>(),
        1..=3_u32,
        1..=2_u32,
    )
        .prop_map(|(origin, depth, mips, layers)| TextureRecipe {
            origin,
            depth,
            mips,
            layers,
        });
    let used = (
        any::<usize>(),
        0..3_u32,
        0..2_u32,
        0..3_u8,
        any::<bool>(),
        any::<u8>(),
    )
        .prop_map(|(resource, mip, layer, part, write, variant)| UseRecipe {
            resource,
            mip,
            layer,
            part,
            write,
            variant,
        });
    let kind = prop_oneof![
        Just(Kind::Transfer),
        Just(Kind::Compute),
        Just(Kind::Raster),
        Just(Kind::Chain),
    ];
    let pass = (
        kind,
        prop::collection::vec(used, 1..=4),
        any::<bool>(),
        any::<u8>(),
    )
        .prop_map(|(kind, uses, side_effects, queue)| PassRecipe {
            kind,
            uses,
            side_effects,
            queue,
        });
    (
        prop::collection::vec(texture, 1..=4),
        prop::collection::vec(origin(BUFFER_IMPORTS.len()), 0..=3),
        prop::collection::vec(pass, 1..=10),
    )
        .prop_map(|(textures, buffers, passes)| GraphRecipe {
            textures,
            buffers,
            passes,
        })
}

/// A texture of a graph being built.
#[derive(Debug)]
struct Texture {
    /// Its ID.
    id: TextureId,
    /// Whether it's transient, so a read needs an earlier write.
    transient: bool,
    /// Whether it's imported, so its shape is the caller's.
    imported: bool,
    /// Whether it's a depth texture.
    depth: bool,
    /// Its mips, or as many as uses of an import name.
    mips: u32,
    /// Its layers, or as many as uses of an import name.
    layers: u32,
    /// Whether each mip of each layer is written so far, by
    /// `mip * layers + layer`.
    written: Vec<bool>,
}

impl Texture {
    /// The part of it `used` names, and that part's cells. If `one_mip`,
    /// it's one mip, as a storage view is, rather than all of them.
    fn part(&self, used: &UseRecipe, one_mip: bool) -> (TextureUse, Vec<usize>) {
        let mip = used.mip % self.mips;
        let layer = used.layer % self.layers;
        let part = if one_mip { used.part.max(1) } else { used.part };
        let (mips, layers) = match part {
            0 => (0..self.mips, 0..self.layers),
            1 => (mip..mip + 1, 0..self.layers),
            _ => (mip..mip + 1, layer..layer + 1),
        };
        let cells = mips
            .clone()
            .flat_map(|mip| layers.clone().map(move |layer| self.cell(mip, layer)))
            .collect();
        let subresource = match part {
            0 => Subresource::ALL,
            1 => Subresource {
                mips,
                layers: 0..Subresource::REMAINING,
            },
            _ => Subresource { mips, layers },
        };
        (self.id.subresource(subresource), cells)
    }

    /// The index of mip `mip` of layer `layer` in `written`.
    fn cell(&self, mip: u32, layer: u32) -> usize {
        (mip * self.layers + layer) as usize
    }

    /// Whether every one of `cells` has contents to read: written, or from
    /// before the frame.
    fn readable(&self, cells: &[usize]) -> bool {
        !self.transient || cells.iter().all(|&cell| self.written[cell])
    }
}

/// A buffer of a graph being built.
#[derive(Debug)]
struct Buffer {
    /// Its ID.
    id: BufferId,
    /// Whether it has contents to read: written, or from before the frame.
    readable: bool,
}

/// A use fitted to its pass, ready to declare.
enum Fitted {
    /// Read part of a texture.
    ReadTexture(TextureUse, AccessSet),
    /// Write part of a texture, whose cells these are.
    WriteTexture(TextureUse, AccessSet, usize, Vec<usize>),
    /// Read a buffer.
    ReadBuffer(BufferId, AccessSet),
    /// Write a buffer, at this index of the graph's.
    WriteBuffer(BufferId, AccessSet, usize),
    /// Render to mip `mip` of layer `layer` of the texture at this index.
    Colour(usize, u32, u32, Attachment<[f32; 4]>),
    /// Depth-test against mip `mip` of layer `layer` of the texture at this
    /// index.
    Depth(usize, u32, u32, Attachment<f32>),
}

/// The description `recipe` makes, finished with a pass that writes the
/// output.
fn build(recipe: &GraphRecipe) -> GraphDesc {
    let mut graph = GraphDesc::new("random");
    let mut textures: Vec<_> = recipe
        .textures
        .iter()
        .enumerate()
        .map(|(index, texture)| {
            let name = format!("texture {index}");
            let format = if texture.depth {
                Format::Depth32Float
            } else {
                Format::Rgba16Float
            };
            let desc = TransientTexture {
                mips: MipCount::Count(texture.mips),
                layers: texture.layers,
                ..TransientTexture::new(format, TextureSize::Fixed(Extent3d::new_2d(64, 64)))
            };
            let (id, imported) = match texture.origin {
                Origin::Transient => (graph.transient_texture(&name, desc), false),
                Origin::Persistent => (graph.persistent_texture(&name, desc), false),
                Origin::Imported { before, after } => {
                    let import = ImportDesc {
                        before: TEXTURE_IMPORTS[before].into(),
                        after: TEXTURE_IMPORTS[after].into(),
                    };
                    (graph.import_texture(&name, import), true)
                }
            };
            let (mips, layers) = if imported {
                (IMPORT_EXTENT, IMPORT_EXTENT)
            } else {
                (texture.mips, texture.layers)
            };
            Texture {
                id,
                transient: matches!(texture.origin, Origin::Transient),
                imported,
                depth: texture.depth && !imported,
                mips,
                layers,
                written: vec![false; (mips * layers) as usize],
            }
        })
        .collect();
    let mut buffers: Vec<_> = recipe
        .buffers
        .iter()
        .enumerate()
        .map(|(index, &origin)| {
            let name = format!("buffer {index}");
            let id = match origin {
                Origin::Transient => graph.transient_buffer(&name, 256),
                Origin::Persistent => graph.persistent_buffer(&name, 256),
                Origin::Imported { before, after } => graph.import_buffer(
                    &name,
                    ImportDesc {
                        before: BUFFER_IMPORTS[before].into(),
                        after: BUFFER_IMPORTS[after].into(),
                    },
                ),
            };
            Buffer {
                id,
                readable: !matches!(origin, Origin::Transient),
            }
        })
        .collect();

    for (index, pass) in recipe.passes.iter().enumerate() {
        let name = format!("pass {index}");
        if let Kind::Chain = pass.kind {
            chain(&mut graph, &name, pass, &mut textures, &buffers);
            continue;
        }
        let fitted = fit(pass, &textures, &buffers);
        let renders = fitted
            .iter()
            .any(|used| matches!(used, Fitted::Colour(..) | Fitted::Depth(..)));
        let writes = fitted
            .iter()
            .any(|used| !matches!(used, Fitted::ReadTexture(..) | Fitted::ReadBuffer(..)));
        let declared = match pass.kind {
            Kind::Raster if renders => {
                declare_raster(&mut graph, &name, pass, &fitted, &textures);
                true
            }
            Kind::Compute if !fitted.is_empty() => {
                let mut builder = graph.compute_pass(&name, ComputeRecorder::recorder("compute"));
                if pass.queue % 2 == 1 {
                    builder.queue(QueueKind::Compute);
                }
                if pass.side_effects || !writes {
                    builder.side_effects();
                }
                for used in &fitted {
                    match used {
                        Fitted::ReadTexture(part, access) => builder.read(part.clone(), *access),
                        Fitted::WriteTexture(part, access, ..) => {
                            builder.write(part.clone(), *access)
                        }
                        Fitted::ReadBuffer(buffer, access) => builder.read(*buffer, *access),
                        Fitted::WriteBuffer(buffer, access, _) => builder.write(*buffer, *access),
                        Fitted::Colour(..) | Fitted::Depth(..) => {
                            unreachable!("only raster renders")
                        }
                    };
                }
                true
            }
            Kind::Transfer if !fitted.is_empty() => {
                let mut builder = graph.transfer_pass(&name, "transfer");
                builder.queue(
                    [QueueKind::Graphics, QueueKind::Compute, QueueKind::Transfer]
                        [usize::from(pass.queue % 3)],
                );
                if pass.side_effects || !writes {
                    builder.side_effects();
                }
                for used in &fitted {
                    match used {
                        Fitted::ReadTexture(part, access) => builder.read(part.clone(), *access),
                        Fitted::WriteTexture(part, access, ..) => {
                            builder.write(part.clone(), *access)
                        }
                        Fitted::ReadBuffer(buffer, access) => builder.read(*buffer, *access),
                        Fitted::WriteBuffer(buffer, access, _) => builder.write(*buffer, *access),
                        Fitted::Colour(..) | Fitted::Depth(..) => {
                            unreachable!("only raster renders")
                        }
                    };
                }
                true
            }
            _ => false,
        };
        if declared {
            mark(&fitted, &mut textures, &mut buffers);
        }
    }
    finished(&graph)
}

/// `pass`'s uses fitted to its kind: at most one of each resource, a read
/// of contents no earlier pass wrote made a write where the kind can
/// write, and otherwise left out. A raster pass renders to the textures
/// it writes, all at one mip, so they're the same size, and reads the
/// rest.
fn fit(pass: &PassRecipe, textures: &[Texture], buffers: &[Buffer]) -> Vec<Fitted> {
    let mut fitted = Vec::new();
    let mut named = Vec::new();
    let mut target_mip = None;
    let (mut colours, mut depth) = (0, false);
    let resources = textures.len() + buffers.len();
    for used in &pass.uses {
        let resource = used.resource % resources;
        if named.contains(&resource) {
            continue;
        }
        named.push(resource);
        let variant = usize::from(used.variant);
        if let Some(texture) = textures.get(resource) {
            let (part, cells) = texture.part(used, matches!(pass.kind, Kind::Compute));
            let readable = texture.readable(&cells);
            let fits = match pass.kind {
                Kind::Transfer => match (used.write, readable) {
                    (false, true) => Some(Fitted::ReadTexture(part, Access::CopyRead.into())),
                    _ => Some(Fitted::WriteTexture(
                        part,
                        Access::CopyWrite.into(),
                        resource,
                        cells,
                    )),
                },
                Kind::Compute => {
                    let read = if texture.depth || variant % 2 == 0 {
                        Access::ComputeShaderSampledRead
                    } else {
                        Access::ComputeShaderRead
                    };
                    match (used.write && !texture.depth, readable) {
                        (true, true) if variant % 2 == 1 => Some(Fitted::WriteTexture(
                            part,
                            Access::ComputeShaderRead | Access::ComputeShaderWrite,
                            resource,
                            cells,
                        )),
                        (false, true) => Some(Fitted::ReadTexture(part, read.into())),
                        (_, _) if !texture.depth => Some(Fitted::WriteTexture(
                            part,
                            Access::ComputeShaderWrite.into(),
                            resource,
                            cells,
                        )),
                        _ => None,
                    }
                }
                Kind::Raster if used.write && !texture.imported => {
                    let mip = *target_mip.get_or_insert(used.mip % texture.mips);
                    let layer = used.layer % texture.layers;
                    let cell_readable =
                        mip < texture.mips && texture.readable(&[texture.cell(mip, layer)]);
                    if mip >= texture.mips {
                        None
                    } else if texture.depth && !depth {
                        depth = true;
                        let attachment = match variant % 3 {
                            1 if cell_readable => Attachment::Keep,
                            2 => Attachment::Discard,
                            _ => Attachment::Clear(0.0),
                        };
                        Some(Fitted::Depth(resource, mip, layer, attachment))
                    } else if !texture.depth && colours < 2 {
                        colours += 1;
                        let attachment = match variant % 3 {
                            1 if cell_readable => Attachment::Keep,
                            2 => Attachment::Discard,
                            _ => Attachment::Clear([0.0; 4]),
                        };
                        Some(Fitted::Colour(resource, mip, layer, attachment))
                    } else {
                        None
                    }
                }
                Kind::Raster => readable
                    .then(|| Fitted::ReadTexture(part, Access::FragmentShaderSampledRead.into())),
                Kind::Chain => unreachable!("a chain is fitted on its own"),
            };
            fitted.extend(fits);
        } else {
            let index = resource - textures.len();
            let buffer = &buffers[index];
            let id = buffer.id;
            let fits = match pass.kind {
                Kind::Transfer => match (used.write, buffer.readable) {
                    (false, true) => Some(Fitted::ReadBuffer(id, Access::CopyRead.into())),
                    _ => Some(Fitted::WriteBuffer(id, Access::CopyWrite.into(), index)),
                },
                Kind::Compute => match (used.write, buffer.readable) {
                    (false, true) => {
                        let read = [Access::ComputeShaderRead, Access::IndirectRead][variant % 2];
                        Some(Fitted::ReadBuffer(id, read.into()))
                    }
                    (true, true) if variant % 2 == 1 => Some(Fitted::WriteBuffer(
                        id,
                        Access::ComputeShaderRead | Access::ComputeShaderWrite,
                        index,
                    )),
                    _ => Some(Fitted::WriteBuffer(
                        id,
                        Access::ComputeShaderWrite.into(),
                        index,
                    )),
                },
                Kind::Raster => buffer.readable.then(|| {
                    let read = [
                        Access::FragmentShaderRead,
                        Access::VertexShaderRead,
                        Access::IndexRead,
                        Access::IndirectRead,
                    ][variant % 4];
                    Fitted::ReadBuffer(id, read.into())
                }),
                Kind::Chain => unreachable!("a chain is fitted on its own"),
            };
            fitted.extend(fits);
        }
    }
    fitted
}

/// Declares a raster pass named `name`, with `fitted`'s uses, of
/// `textures`.
fn declare_raster(
    graph: &mut GraphDesc,
    name: &str,
    pass: &PassRecipe,
    fitted: &[Fitted],
    textures: &[Texture],
) {
    let mut builder = graph.raster_pass(name, "raster");
    if pass.side_effects {
        builder.side_effects();
    }
    for used in fitted {
        match used {
            Fitted::ReadTexture(part, access) => builder.read(part.clone(), *access),
            Fitted::ReadBuffer(buffer, access) => builder.read(*buffer, *access),
            Fitted::Colour(texture, mip, layer, attachment) => {
                builder.colour(textures[*texture].id.attachment(*mip, *layer), *attachment)
            }
            Fitted::Depth(texture, mip, layer, attachment) => {
                builder.depth(textures[*texture].id.attachment(*mip, *layer), *attachment)
            }
            Fitted::WriteTexture(..) | Fitted::WriteBuffer(..) => {
                unreachable!("a raster pass writes only its attachments")
            }
        };
    }
}

/// Records what `fitted`'s uses write in `textures` and `buffers`.
fn mark(fitted: &[Fitted], textures: &mut [Texture], buffers: &mut [Buffer]) {
    for used in fitted {
        match used {
            Fitted::WriteTexture(_, _, texture, cells) => {
                for &cell in cells {
                    textures[*texture].written[cell] = true;
                }
            }
            Fitted::Colour(texture, mip, layer, _) | Fitted::Depth(texture, mip, layer, _) => {
                let texture = &mut textures[*texture];
                let cell = texture.cell(*mip, *layer);
                texture.written[cell] = true;
            }
            Fitted::WriteBuffer(_, _, buffer) => buffers[*buffer].readable = true,
            Fitted::ReadTexture(..) | Fitted::ReadBuffer(..) => {}
        }
    }
}

/// Declares a compute pass named `name`, repeated for each mip of the first
/// colour texture `pass`'s uses name that the graph creates with more than
/// one mip, if there's one whose first previous mip has contents, reading
/// the previous mip and writing the current one, and reading what the
/// other uses name, where it has contents.
fn chain(
    graph: &mut GraphDesc,
    name: &str,
    pass: &PassRecipe,
    textures: &mut [Texture],
    buffers: &[Buffer],
) {
    let resources = textures.len() + buffers.len();
    let Some(first) = pass.uses.iter().find(|used| {
        textures
            .get(used.resource % resources)
            .is_some_and(|texture| !texture.imported && !texture.depth && texture.mips > 1)
    }) else {
        return;
    };
    let index = first.resource % resources;
    let texture = &textures[index];
    let (direction, start) = if first.variant % 2 == 0 {
        (MipDirection::Down, 0)
    } else {
        (MipDirection::Up, texture.mips - 1)
    };
    let previous: Vec<_> = (0..texture.layers)
        .map(|layer| texture.cell(start, layer))
        .collect();
    if !texture.readable(&previous) {
        return;
    }
    let id = texture.id;
    let mut builder = graph.compute_pass(name, ComputeRecorder::dispatch("chain"));
    builder
        .for_each_mip(id, direction, ..)
        .bind(id.previous_mip(), Access::ComputeShaderSampledRead)
        .bind(id.current_mip(), Access::ComputeShaderWrite);
    if pass.side_effects {
        builder.side_effects();
    }
    let mut named = vec![index];
    for used in &pass.uses {
        let resource = used.resource % resources;
        if named.contains(&resource) {
            continue;
        }
        named.push(resource);
        if let Some(other) = textures.get(resource) {
            let (part, cells) = other.part(used, false);
            if other.readable(&cells) {
                builder.bind(part, Access::ComputeShaderSampledRead);
            }
        } else {
            let buffer = &buffers[resource - textures.len()];
            if buffer.readable {
                builder.bind(buffer.id, Access::ComputeShaderRead);
            }
        }
    }
    let texture = &mut textures[index];
    for mip in (0..texture.mips).filter(|&mip| mip != start) {
        for layer in 0..texture.layers {
            let cell = texture.cell(mip, layer);
            texture.written[cell] = true;
        }
    }
}

/// A valid schedule of the passes `kept` keeps, of `passes`, whose reduced
/// dependency graph is `dependencies`, picked by `choices`, used in turn and
/// over again: each picks the next pass from those ready to run, and
/// whether it joins the group before it, where it can.
fn random_order(
    choices: &[u32],
    passes: &[ExpandedPass],
    dependencies: &Dependencies,
    kept: &[bool],
) -> Scheduled {
    let mut choices = choices.iter().copied().cycle();
    let mut placed = vec![false; kept.len()];
    let count = kept.iter().filter(|&&kept| kept).count();
    let mut order: Vec<usize> = Vec::new();
    let mut groups = Vec::new();
    let mut start = 0;
    while order.len() < count {
        let choice = choices.next().expect("there's at least one choice") as usize;
        let ready: Vec<_> = (0..kept.len())
            .filter(|&pass| {
                kept[pass]
                    && !placed[pass]
                    && dependencies.on[pass]
                        .iter()
                        .all(|dependency| placed[dependency.on])
            })
            .collect();
        let pass = ready[choice % ready.len()];
        let group = &order[start..];
        let joins = !group.is_empty()
            && (choice >> 16) % 2 == 1
            && group.iter().all(|&member| {
                dependencies.on[pass]
                    .iter()
                    .all(|dependency| dependency.on != member)
                    && layout_clash(&passes[member], &passes[pass]).is_none()
            });
        if !joins && !group.is_empty() {
            groups.push(start..order.len());
            start = order.len();
        }
        order.push(pass);
        placed[pass] = true;
    }
    if start < order.len() {
        groups.push(start..order.len());
    }
    Scheduled {
        passes: order,
        groups,
    }
}

/// The plan of `desc`, a valid description, compiled in the order
/// `choices` picks.
fn compiled_in_order(desc: &GraphDesc, choices: &[u32]) -> Plan {
    let plan = compile_in_order(desc, &inputs(), |passes, dependencies, kept| {
        random_order(choices, passes, dependencies, kept)
    });
    plan.unwrap_or_else(|errors| panic!("the graph is valid: {errors:?}"))
}

/// The messages of the hazards `check` finds in `plan`, of `desc`.
fn hazards(desc: &GraphDesc, plan: &Plan) -> Vec<String> {
    match check(desc, plan) {
        Ok(()) => Vec::new(),
        Err(hazards) => hazards
            .iter()
            .map(|hazard| hazard.message().to_owned())
            .collect(),
    }
}

proptest! {
    /// Every random graph compiles, under the schedule, to a plan the
    /// checker finds no hazards in.
    #[test]
    fn plans_of_random_graphs_have_no_hazards(recipe in graphs()) {
        let desc = build(&recipe);
        let plan = compile(&desc, &inputs())
            .unwrap_or_else(|errors| panic!("the graph is valid: {errors:?}"));
        prop_assert_eq!(hazards(&desc, &plan), Vec::<String>::new(), "{}", plan.describe());
    }

    /// Every random graph compiles to a plan with no hazards in any valid
    /// order and grouping of its passes, so barrier derivation is right for
    /// any order a schedule might choose.
    #[test]
    fn random_graphs_in_random_orders_have_no_hazards(
        recipe in graphs(),
        choices in prop::collection::vec(any::<u32>(), 1..32),
    ) {
        let desc = build(&recipe);
        let plan = compiled_in_order(&desc, &choices);
        prop_assert_eq!(hazards(&desc, &plan), Vec::<String>::new(), "{}", plan.describe());
    }

    /// The schedule of a random graph is valid, and the same on every run.
    #[test]
    fn schedules_of_random_graphs_are_valid_and_repeatable(recipe in graphs()) {
        let desc = build(&recipe);
        let inputs = inputs();
        let valid = |errors| panic!("the graph is valid: {errors:?}");
        let checked = compile_in_order(&desc, &inputs, |_, _, kept| {
            schedule(Schedule::AsWritten, kept)
        })
        .unwrap_or_else(valid);
        prop_assert_eq!(checked, compile(&desc, &inputs).unwrap_or_else(valid));
    }

    /// Every barrier of a random graph's plan, in a random order, is
    /// needed: the checker finds a hazard in the plan without it. There are
    /// four exceptions, where the compiler records barriers it could
    /// leave out, as `TODO.md` says:
    ///
    /// - one on an import that covers none of the mips and layers the
    ///   passes name orders nothing;
    /// - a global barrier orders every buffer, so where a plan has more
    ///   than one, another may order what it does;
    /// - one at the end of the frame in one layout, or of buffers, which
    ///   have none, whose writes are among
    ///   the accesses it leaves, may be covered by the caller's own
    ///   ordering, after the stages of those accesses, if the accesses
    ///   before it are in those stages, which only the RHI knows;
    /// - one from the last frame's reads to the same reads, of a persistent
    ///   texture, orders nothing, since before the first frame the executor
    ///   moves the texture to the accesses the frame ends with, but the
    ///   checker starts that frame from `Access::None`, and needs it.
    #[test]
    fn the_checker_finds_any_missing_barrier(
        recipe in graphs(),
        choices in prop::collection::vec(any::<u32>(), 1..32),
        barrier in any::<usize>(),
    ) {
        let desc = build(&recipe);
        let plan = compiled_in_order(&desc, &choices);
        let global = |barrier: &PlannedBarrier| barrier.target == BarrierTarget::Global;
        let globals = plan.barriers.iter().filter(|&barrier| global(barrier)).count();
        let covering: Vec<_> = (0..plan.barriers.len())
            .filter(|&index| covers_anything(&plan, index))
            .filter(|&index| globals == 1 || !global(&plan.barriers[index]))
            .filter(|&index| {
                let barrier = &plan.barriers[index];
                let texture = matches!(barrier.target, BarrierTarget::Texture(..));
                let laid_out = !texture || barrier.before.layout() == barrier.after.layout();
                let covered = laid_out
                    && barrier
                        .before
                        .iter()
                        .filter(|access| access.is_write())
                        .all(|write| barrier.after.contains(write));
                !(plan.end_of_frame.contains(&index) && covered)
            })
            .filter(|&index| {
                let barrier = &plan.barriers[index];
                !(barrier.from_last_frame && barrier.before == barrier.after)
            })
            .collect();
        if !covering.is_empty() {
            let index = covering[barrier % covering.len()];
            prop_assert!(
                !hazards(&desc, &plan.without_barrier(index)).is_empty(),
                "barrier {} of:\n{}",
                index,
                plan.describe()
            );
        }
    }
}
