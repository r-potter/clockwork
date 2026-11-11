//! Render state: the rasterization state draws use, which programs don't
//! hold, so it's set in a command list (design 0002, "Programs").

/// The most colour attachments a [`RenderState`] describes, numbered from 0.
/// Rendering may use fewer, as the device allows.
pub const MAX_COLOUR_ATTACHMENTS: usize = 8;

/// The rasterization state draws use, set with
/// [`CommandList::set_render_state`](crate::CommandList::set_render_state)
/// (design 0002, "Programs"). Programs hold no state of their own, so the
/// same program draws with any render state.
///
/// [`Default`] gives the project's conventions: triangle lists,
/// counter-clockwise front faces, back faces culled, a reverse-Z depth test
/// ([`CompareOp::Greater`]) that writes depth, and every colour attachment
/// written in full. Change it with the methods, which
/// chain:
///
/// ```
/// # use clockwork_rhi::{CullMode, RenderState};
/// let state = RenderState::default().cull(CullMode::None).depth(None);
/// ```
///
/// Stencil tests are always off. Polygons are filled, with one sample per
/// pixel: line and point fill modes need a device feature, and sample counts
/// need multisampled textures, so each arrives with the code that needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderState {
    /// How vertices form primitives.
    pub(crate) topology: Topology,
    /// Which faces are culled.
    pub(crate) cull: CullMode,
    /// Which winding is a front face.
    pub(crate) front_face: FrontFace,
    /// The depth test, or `None` for none.
    pub(crate) depth: Option<DepthTest>,
    /// How each colour attachment is written, by its number.
    pub(crate) colour: [ColourWrite; MAX_COLOUR_ATTACHMENTS],
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            topology: Topology::TriangleList,
            cull: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            depth: Some(DepthTest {
                compare: CompareOp::Greater,
                write: true,
            }),
            colour: [ColourWrite::default(); MAX_COLOUR_ATTACHMENTS],
        }
    }
}

impl RenderState {
    /// This state, with vertices forming primitives as `topology`.
    #[must_use]
    pub const fn topology(self, topology: Topology) -> Self {
        Self { topology, ..self }
    }

    /// This state, culling `cull`.
    #[must_use]
    pub const fn cull(self, cull: CullMode) -> Self {
        Self { cull, ..self }
    }

    /// This state, with front faces wound as `front_face`.
    #[must_use]
    pub const fn front_face(self, front_face: FrontFace) -> Self {
        Self { front_face, ..self }
    }

    /// This state, with the depth test `depth`, or with none for `None`, so
    /// that depth is neither tested nor written.
    #[must_use]
    pub const fn depth(self, depth: Option<DepthTest>) -> Self {
        Self { depth, ..self }
    }

    /// This state, writing the colour attachment numbered `attachment` as
    /// `write` says.
    ///
    /// # Panics
    ///
    /// Panics if `attachment` isn't less than [`MAX_COLOUR_ATTACHMENTS`].
    #[must_use]
    pub const fn colour(self, attachment: usize, write: ColourWrite) -> Self {
        // A const fn can't format the limit into the message, so it's
        // written out, and this keeps it in step with the constant.
        const {
            assert!(
                MAX_COLOUR_ATTACHMENTS == 8,
                "update the panic message below",
            );
        }
        assert!(
            attachment < MAX_COLOUR_ATTACHMENTS,
            "a render state describes colour attachments 0 to 7",
        );
        let mut colour = self.colour;
        colour[attachment] = write;
        Self { colour, ..self }
    }
}

/// How vertices form primitives. Strips never restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    /// Each vertex is a point.
    PointList,
    /// Each pair of vertices is a line.
    LineList,
    /// Each vertex after the first ends a line from the one before.
    LineStrip,
    /// Each three vertices are a triangle.
    TriangleList,
    /// Each vertex after the second ends a triangle with the two before.
    TriangleStrip,
}

/// Which faces of triangles are culled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// None: both faces are drawn.
    None,
    /// Front faces.
    Front,
    /// Back faces.
    Back,
}

/// Which winding of a triangle's vertices, as seen on screen, makes it a
/// front face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Counter-clockwise, the project's convention.
    CounterClockwise,
    /// Clockwise.
    Clockwise,
}

/// A depth test: how a fragment's depth is compared with the attachment's,
/// and whether a fragment that passes writes its depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthTest {
    /// How the fragment's depth is compared with the attachment's. With
    /// reverse-Z, nearer is greater.
    pub compare: CompareOp,
    /// Whether a fragment that passes writes its depth.
    pub write: bool,
}

/// How a value is compared with a reference: the depth test passes if the
/// fragment's depth is `op` the attachment's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Never passes.
    Never,
    /// Passes if less.
    Less,
    /// Passes if equal.
    Equal,
    /// Passes if less or equal.
    LessOrEqual,
    /// Passes if greater, the reverse-Z depth test.
    Greater,
    /// Passes if not equal.
    NotEqual,
    /// Passes if greater or equal.
    GreaterOrEqual,
    /// Always passes.
    Always,
}

/// How a colour attachment is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColourWrite {
    /// Which channels are written.
    pub mask: ColourMask,
}

impl Default for ColourWrite {
    /// Every channel written.
    fn default() -> Self {
        Self {
            mask: ColourMask::ALL,
        }
    }
}

/// Which channels of a colour attachment are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColourMask {
    /// Red.
    pub red: bool,
    /// Green.
    pub green: bool,
    /// Blue.
    pub blue: bool,
    /// Alpha.
    pub alpha: bool,
}

impl ColourMask {
    /// Every channel.
    pub const ALL: Self = Self {
        red: true,
        green: true,
        blue: true,
        alpha: true,
    };
    /// No channel, so nothing is written.
    pub const NONE: Self = Self {
        red: false,
        green: false,
        blue: false,
        alpha: false,
    };
}

// Render state is plain data, recorded on any thread (design 0002,
// "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RenderState>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of render state's defaults and methods.

    use super::*;

    /// The defaults are the project's conventions (CLAUDE.md, "Rendering"):
    /// counter-clockwise front faces, back faces culled, and a reverse-Z
    /// depth test that writes, with every colour attachment written in full.
    #[test]
    fn defaults_are_the_project_conventions() {
        let state = RenderState::default();
        assert_eq!(state.topology, Topology::TriangleList);
        assert_eq!(state.cull, CullMode::Back);
        assert_eq!(state.front_face, FrontFace::CounterClockwise);
        assert_eq!(
            state.depth,
            Some(DepthTest {
                compare: CompareOp::Greater,
                write: true,
            }),
        );
        for write in state.colour {
            assert_eq!(write.mask, ColourMask::ALL);
        }
    }

    /// Each method changes its own state and leaves the rest.
    #[test]
    fn methods_change_only_their_state() {
        let default = RenderState::default();
        let masked = ColourWrite {
            mask: ColourMask::NONE,
        };
        let state = default
            .topology(Topology::LineStrip)
            .cull(CullMode::None)
            .front_face(FrontFace::Clockwise)
            .depth(None)
            .colour(3, masked);
        assert_eq!(state.topology, Topology::LineStrip);
        assert_eq!(state.cull, CullMode::None);
        assert_eq!(state.front_face, FrontFace::Clockwise);
        assert_eq!(state.depth, None);
        let mut colour = default.colour;
        colour[3] = masked;
        assert_eq!(state.colour, colour);
    }

    /// Describing a colour attachment past the last panics.
    #[test]
    #[should_panic(expected = "a render state describes colour attachments 0 to 7")]
    fn colour_past_the_last_attachment_panics() {
        let _ = RenderState::default().colour(MAX_COLOUR_ATTACHMENTS, ColourWrite::default());
    }
}
