//! The Khronos validation layer: whether and how to enable it, and the debug
//! messenger callback that turns its messages into `tracing` events.
//!
//! [`decide`] turns the requested [`Validation`] and what the Vulkan loader
//! offers ([`Availability`]) into a [`Plan`], or an error if validation is
//! required but can't be had (R7). It's a pure function, so every case is
//! tested without a GPU. The instance is then created with the plan's layer,
//! settings ([`LayerSettings`], through `VK_EXT_layer_settings`) and debug
//! messenger.
//!
//! [`debug_callback`] emits every message on [`VALIDATION_TARGET`], with the
//! backend's session as an explicit field (R5), since the layer may call it on
//! threads with no session span. It does nothing else: classifying and
//! counting messages is done by the `tracing` layers in
//! [`crate::diagnostics`].

use std::ffi::{CStr, c_void};
use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use ash::{Entry, vk};
use tracing::{Level, debug, error, trace, warn};

use crate::diagnostics::{SessionId, VALIDATION_TARGET};
use crate::rhi::{Validation, ValidationFeature, ValidationFeatureSet};

/// The Khronos validation layer, shipped with the Vulkan SDK.
pub const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The validation layer's setting for each feature, set through
/// `VK_EXT_layer_settings`.
///
/// The names are the `key`s of the settings in the layer's manifest,
/// `VkLayer_khronos_validation.json`, as of Vulkan SDK 1.4.363.0. Each is a
/// boolean, and every feature's setting is passed, true or false, so that
/// what's validated doesn't depend on the layer's defaults.
///
/// A layer that doesn't recognise a name (if a later version renames it, for
/// example) reports it during instance creation, as a warning with the ID
/// `VALIDATION-SETTINGS`. That's counted like any other warning, so it fails
/// every T1 test, rather than leaving a feature silently off.
const FEATURE_SETTINGS: [(ValidationFeature, &CStr); 4] = [
    (ValidationFeature::Core, c"validate_core"),
    (ValidationFeature::Sync, c"validate_sync"),
    (ValidationFeature::Gpu, c"gpuav_enable"),
    (ValidationFeature::BestPractices, c"validate_best_practices"),
];

/// What the Vulkan loader offers for validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Availability {
    /// Whether [`VALIDATION_LAYER`] is installed.
    pub layer: bool,
    /// Whether the layer provides `VK_EXT_layer_settings`, which every
    /// feature but [`ValidationFeature::Core`] needs to be enabled.
    pub layer_settings: bool,
}

impl Availability {
    /// Asks the Vulkan loader what's available.
    ///
    /// # Errors
    ///
    /// Returns an error if the instance layers, or the layer's extensions,
    /// can't be enumerated.
    pub fn query(entry: &Entry) -> Result<Self> {
        // SAFETY: `entry` holds a valid, loaded Vulkan loader.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .context("failed to enumerate Vulkan instance layers")?;
        let layer = layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        if !layer {
            return Ok(Self {
                layer,
                layer_settings: false,
            });
        }
        // SAFETY: `entry` holds a valid, loaded Vulkan loader, and the layer
        // name is a valid, null-terminated string.
        let extensions =
            unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
                .context("failed to enumerate the Vulkan validation layer's extensions")?;
        let layer_settings = extensions.iter().any(|extension| {
            extension.extension_name_as_c_str() == Ok(vk::EXT_LAYER_SETTINGS_NAME)
        });
        Ok(Self {
            layer,
            layer_settings,
        })
    }
}

/// How the instance is created for validation, decided by [`decide`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The features to enable. [`ValidationFeatureSet::OFF`] means no layer and
    /// no debug messenger.
    pub features: ValidationFeatureSet,
    /// Whether to pass the features to the layer as settings, through
    /// `VK_EXT_layer_settings`. Without it, the layer runs with its defaults,
    /// which are core validation only.
    pub use_settings: bool,
    /// A warning to log, when validation was preferred and less than was
    /// asked for is available.
    pub warning: Option<String>,
}

impl Plan {
    /// A plan with no validation and nothing to warn about.
    const OFF: Self = Self {
        features: ValidationFeatureSet::OFF,
        use_settings: false,
        warning: None,
    };
}

/// Decides how to enable the `requested` validation with what's `available`
/// (R7). Pure.
///
/// - `Off`, or no features: no validation.
/// - The layer isn't installed: an error if validation is required.
///   Otherwise, no validation, with a warning.
/// - Features other than core are requested, but the layer doesn't provide
///   `VK_EXT_layer_settings`: an error if validation is required. Otherwise,
///   core validation only, with a warning.
/// - Otherwise, every requested feature, through the layer's settings.
///
/// # Errors
///
/// Returns an error, naming the layer or the features, if validation is
/// required and they aren't available.
pub fn decide(requested: Validation, available: Availability) -> Result<Plan> {
    let (features, required) = match requested {
        Validation::Off => return Ok(Plan::OFF),
        Validation::Preferred(features) => (features, false),
        Validation::Required(features) => (features, true),
    };
    if features.is_off() {
        return Ok(Plan::OFF);
    }
    let layer = VALIDATION_LAYER.to_string_lossy();

    if !available.layer {
        if required {
            bail!("validation ({features}) is required, but {layer} isn't installed");
        }
        return Ok(Plan {
            warning: Some(format!(
                "{layer} isn't installed; continuing without Vulkan validation"
            )),
            ..Plan::OFF
        });
    }

    let needs_settings: Vec<_> = features
        .iter()
        .filter(|&feature| feature != ValidationFeature::Core)
        .map(ValidationFeature::name)
        .collect();
    if !available.layer_settings && !needs_settings.is_empty() {
        let missing = needs_settings.join(", ");
        if required {
            bail!(
                "validation features {missing} are required, but {layer} doesn't provide \
                 VK_EXT_layer_settings to enable them"
            );
        }
        return Ok(Plan {
            features: ValidationFeatureSet::OFF.with(ValidationFeature::Core),
            use_settings: false,
            warning: Some(format!(
                "{layer} doesn't provide VK_EXT_layer_settings, so validation features \
                 {missing} can't be enabled; continuing with core validation only"
            )),
        });
    }

    Ok(Plan {
        features,
        use_settings: available.layer_settings,
        warning: None,
    })
}

/// The validation layer's settings for a set of features, kept together so
/// that the values outlive the [`vk::LayerSettingEXT`]s that point to them.
pub struct LayerSettings {
    /// Each feature's value, in [`FEATURE_SETTINGS`]'s order.
    values: [vk::Bool32; FEATURE_SETTINGS.len()],
}

impl LayerSettings {
    /// The settings that enable exactly `features`.
    pub fn new(features: ValidationFeatureSet) -> Self {
        Self {
            values: FEATURE_SETTINGS
                .map(|(feature, _)| vk::Bool32::from(features.contains(feature))),
        }
    }

    /// The settings, which point into `self`, so borrow it.
    pub fn settings(&self) -> Vec<vk::LayerSettingEXT<'_>> {
        FEATURE_SETTINGS
            .iter()
            .zip(self.values.iter())
            .map(|((_, name), value)| {
                // `LayerSettingEXT::values` takes bytes, and sets the count to
                // the number of bytes, so the value is set by hand: one
                // `VkBool32`.
                vk::LayerSettingEXT {
                    value_count: 1,
                    p_values: std::ptr::from_ref(value).cast(),
                    ..vk::LayerSettingEXT::default()
                        .layer_name(VALIDATION_LAYER)
                        .setting_name(name)
                        .ty(vk::LayerSettingTypeEXT::BOOL32)
                }
            })
            .collect()
    }
}

/// The debug messenger's settings: which messages it reports, the callback,
/// and the session, passed to the callback as its user data.
///
/// Errors and warnings are always requested, whatever the log level, since
/// they're counted and fail tests whatever `RUST_LOG` says. Vulkan's `INFO`
/// and `VERBOSE` messages are requested only if they would be logged.
pub fn messenger_create_info(session: SessionId) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    let mut message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    if tracing::enabled!(target: VALIDATION_TARGET, Level::DEBUG) {
        message_severity |= vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
    }
    if tracing::enabled!(target: VALIDATION_TARGET, Level::TRACE) {
        message_severity |= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
    }
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(message_severity)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
        // The ID is passed as an integer, not a pointer to anything, so
        // nothing is borrowed and there's no lifetime to manage.
        .user_data(session_to_user_data(session))
}

/// Encodes a session ID as the debug messenger's user data.
fn session_to_user_data(session: SessionId) -> *mut c_void {
    // Session IDs come from a counter, so they fit in a pointer on every
    // supported (64-bit) platform.
    std::ptr::without_provenance_mut(session.as_u64() as usize)
}

/// Decodes the session ID that [`session_to_user_data`] encoded.
fn session_from_user_data(user_data: *mut c_void) -> u64 {
    user_data.addr() as u64
}

/// Emits a message from the Vulkan debug messenger as a `tracing` event on
/// [`VALIDATION_TARGET`].
///
/// Vulkan severities map to `tracing` levels as follows:
///
/// | Vulkan    | `tracing` |
/// |-----------|-----------|
/// | `ERROR`   | `ERROR`   |
/// | `WARNING` | `WARN`    |
/// | `INFO`    | `DEBUG`   |
/// | `VERBOSE` | `TRACE`   |
///
/// Vulkan's `INFO` messages are mostly loader and layer diagnostics, so they
/// are logged at `DEBUG` to keep them out of the default `INFO` output.
///
/// The event's fields are `session` (from `user_data`, see
/// [`messenger_create_info`]), `message_id` (`pMessageIdName`, such as a
/// VUID, when there is one) and `types`, and its message is the message's
/// text.
///
/// It never fails or panics: it's called from C, on any thread.
///
/// # Safety
///
/// Called only by the Vulkan implementation, which guarantees that
/// `callback_data` is null or points to valid data for the duration of the
/// call.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: Guaranteed by the caller; see the function's safety section.
    let data = unsafe { callback_data.as_ref() };
    // SAFETY: `data` is valid for the call, as above, and its strings are
    // null or valid, null-terminated strings, as the specification requires.
    let (message, message_id) = unsafe {
        (
            data.and_then(|data| data.message_as_c_str()),
            data.and_then(|data| data.message_id_name_as_c_str()),
        )
    };
    let message = message.map_or("<no message>".into(), CStr::to_string_lossy);
    let message_id = message_id.map(CStr::to_string_lossy);
    let message_id = message_id.as_deref();
    let session = session_from_user_data(user_data);

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(target: VALIDATION_TARGET, session, message_id, ?types, "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: VALIDATION_TARGET, session, message_id, ?types, "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        debug!(target: VALIDATION_TARGET, session, message_id, ?types, "{message}");
    } else {
        trace!(target: VALIDATION_TARGET, session, message_id, ?types, "{message}");
    }
    // Returning `FALSE` lets the Vulkan call that triggered the message
    // continue, as the specification requires.
    vk::FALSE
}

/// Describes what a plan enables, for the log: its features, such as
/// `core,sync`, and whether they rely on the layer's default settings.
pub fn describe(plan: &Plan) -> String {
    let mut description = plan.features.to_string();
    if !plan.use_settings {
        // Writing to a `String` can't fail.
        let _ = write!(description, ", with the layer's default settings");
    }
    description
}

#[cfg(test)]
mod tests {
    //! T0 tests of the validation decision (R7) and the messenger's session
    //! encoding.

    use super::*;

    /// Everything is available.
    const EVERYTHING: Availability = Availability {
        layer: true,
        layer_settings: true,
    };

    /// The layer is installed, but too old to take settings.
    const NO_SETTINGS: Availability = Availability {
        layer: true,
        layer_settings: false,
    };

    /// The layer isn't installed.
    const NO_LAYER: Availability = Availability {
        layer: false,
        layer_settings: false,
    };

    /// Core validation only.
    const CORE: ValidationFeatureSet = ValidationFeatureSet::OFF.with(ValidationFeature::Core);

    /// `Off` needs nothing, and enables nothing, whatever is available.
    #[test]
    fn off_enables_nothing() {
        for available in [EVERYTHING, NO_SETTINGS, NO_LAYER] {
            assert_eq!(decide(Validation::Off, available).unwrap(), Plan::OFF);
        }
    }

    /// Requiring or preferring no features is the same as `Off`.
    #[test]
    fn no_features_enable_nothing() {
        for requested in [
            Validation::Preferred(ValidationFeatureSet::OFF),
            Validation::Required(ValidationFeatureSet::OFF),
        ] {
            assert_eq!(decide(requested, NO_LAYER).unwrap(), Plan::OFF);
        }
    }

    /// With everything available, the features are enabled through the
    /// layer's settings, whether preferred or required.
    #[test]
    fn available_features_are_enabled_through_settings() {
        for requested in [
            Validation::Preferred(ValidationFeatureSet::SYNC),
            Validation::Required(ValidationFeatureSet::SYNC),
        ] {
            assert_eq!(
                decide(requested, EVERYTHING).unwrap(),
                Plan {
                    features: ValidationFeatureSet::SYNC,
                    use_settings: true,
                    warning: None,
                },
            );
        }
    }

    /// Required validation fails if the layer is missing, naming it.
    #[test]
    fn required_validation_fails_without_the_layer() {
        let err = decide(Validation::Required(CORE), NO_LAYER).unwrap_err();
        assert_eq!(
            err.to_string(),
            "validation (core) is required, but VK_LAYER_KHRONOS_validation isn't installed",
        );
    }

    /// Preferred validation continues without the layer, with a warning.
    #[test]
    fn preferred_validation_warns_without_the_layer() {
        let plan = decide(Validation::Preferred(ValidationFeatureSet::SYNC), NO_LAYER).unwrap();
        assert!(plan.features.is_off());
        assert!(plan.warning.unwrap().contains("isn't installed"));
    }

    /// Required features that need settings fail if the layer can't take
    /// them, naming the features.
    #[test]
    fn required_features_fail_without_settings() {
        let features = ValidationFeatureSet::SYNC.with(ValidationFeature::BestPractices);
        let err = decide(Validation::Required(features), NO_SETTINGS).unwrap_err();
        assert!(
            err.to_string()
                .starts_with("validation features sync, best-practices are required"),
            "{err}",
        );
    }

    /// Preferred features that need settings fall back to core validation,
    /// with a warning naming them.
    #[test]
    fn preferred_features_fall_back_to_core_without_settings() {
        let plan = decide(
            Validation::Preferred(ValidationFeatureSet::SYNC),
            NO_SETTINGS,
        )
        .unwrap();
        assert_eq!(plan.features, CORE);
        assert!(!plan.use_settings);
        assert!(plan.warning.unwrap().contains("sync"));
    }

    /// Core validation is the layer's default, so it needs no settings, even
    /// when required.
    #[test]
    fn core_validation_needs_no_settings() {
        let plan = decide(Validation::Required(CORE), NO_SETTINGS).unwrap();
        assert_eq!(plan.features, CORE);
        assert!(!plan.use_settings);
        assert_eq!(plan.warning, None);
    }

    /// Every feature has a setting, and every setting is passed, with the
    /// value saying whether its feature is enabled.
    #[test]
    fn settings_cover_every_feature() {
        let features = ValidationFeatureSet::SYNC;
        let layer_settings = LayerSettings::new(features);
        let settings = layer_settings.settings();
        assert_eq!(settings.len(), ValidationFeature::ALL.len());
        for (setting, (feature, name)) in settings.iter().zip(FEATURE_SETTINGS) {
            // SAFETY: the names point to static strings.
            let setting_name = unsafe { setting.setting_name_as_c_str() };
            assert_eq!(setting_name, Some(name));
            assert_eq!(setting.ty, vk::LayerSettingTypeEXT::BOOL32);
            assert_eq!(setting.value_count, 1);
            // SAFETY: `p_values` points to one `Bool32` in `layer_settings`,
            // which is still alive.
            let value = unsafe { *setting.p_values.cast::<vk::Bool32>() };
            assert_eq!(value == vk::TRUE, features.contains(feature), "{name:?}");
        }
        for feature in ValidationFeature::ALL {
            assert!(
                FEATURE_SETTINGS.iter().any(|&(f, _)| f == feature),
                "{feature:?} has no setting",
            );
        }
    }

    /// A session ID survives the trip through the messenger's user data.
    #[test]
    fn sessions_round_trip_through_user_data() {
        let session = SessionId::unique();
        assert_eq!(
            session_from_user_data(session_to_user_data(session)),
            session.as_u64(),
        );
    }
}
