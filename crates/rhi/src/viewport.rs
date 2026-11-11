//! Viewports and scissors: where draws render to in their attachments, set
//! in a command list (design 0002, "Programs").

/// A size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent2d {
    /// The width, in pixels.
    pub width: u32,
    /// The height, in pixels.
    pub height: u32,
}

/// A position in pixels, from the attachments' top-left corner, with Y
/// pointing down, as in Vulkan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Offset2d {
    /// How far right, in pixels.
    pub x: u32,
    /// How far down, in pixels.
    pub y: u32,
}

/// A rectangle of pixels: `extent` from `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect2d {
    /// The top-left corner.
    pub offset: Offset2d,
    /// The size.
    pub extent: Extent2d,
}

impl Rect2d {
    /// The rectangle of size `extent` from the top-left corner, such as the
    /// whole of an attachment that size.
    #[must_use]
    pub const fn from_extent(extent: Extent2d) -> Self {
        Self {
            offset: Offset2d { x: 0, y: 0 },
            extent,
        }
    }
}

/// How draws map to their attachments: the rectangle normalized device
/// coordinates map to, and the range depths map to.
///
/// The rectangle is in pixels, from the attachments' top-left corner, with Y
/// pointing down, as in Vulkan. It may start off the attachments, and
/// needn't be whole pixels. Depths are from 0 to 1, and `min_depth` may be
/// greater than `max_depth`. Reverse-Z comes from the projection, so a
/// viewport usually keeps the full range, as
/// [`from_extent`](Self::from_extent) does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// The left edge, in pixels.
    pub x: f32,
    /// The top edge, in pixels.
    pub y: f32,
    /// The width, in pixels.
    pub width: f32,
    /// The height, in pixels.
    pub height: f32,
    /// The depth that normalized depth 0 maps to.
    pub min_depth: f32,
    /// The depth that normalized depth 1 maps to.
    pub max_depth: f32,
}

impl Viewport {
    /// The viewport covering the rectangle of size `extent` from the
    /// top-left corner, such as the whole of an attachment that size, with
    /// the full range of depths.
    #[must_use]
    pub const fn from_extent(extent: Extent2d) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the constructors.

    use super::*;

    /// Both constructors cover the extent from the top-left corner, and the
    /// viewport has the full range of depths.
    #[test]
    fn from_extent_covers_the_extent() {
        let extent = Extent2d {
            width: 1920,
            height: 1080,
        };
        assert_eq!(
            Rect2d::from_extent(extent),
            Rect2d {
                offset: Offset2d { x: 0, y: 0 },
                extent,
            },
        );
        assert_eq!(
            Viewport::from_extent(extent),
            Viewport {
                x: 0.0,
                y: 0.0,
                width: 1920.0,
                height: 1080.0,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        );
    }
}
