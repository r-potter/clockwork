//! Streaming: filling buffers and textures from any thread, on the transfer
//! queue (design 0002, "Streaming", R17).
//!
//! An [`Uploader`] stages the bytes it's given in upload buffers of its own,
//! records the copies that move them into place and the barriers around
//! them, and at [`flush`](Uploader::flush) translates and submits them,
//! returning an [`Upload`]: when the copies complete, and the barriers
//! rendering records before it first uses what was uploaded.
//!
//! It's ordinary code over [`GpuResources`] and [`Queue`], outside the
//! `vulkan` module, so it's the same for every backend. Its staging and
//! recording are tested with a fake of [`GpuResources`]. A fake queue can't
//! make a [`CommandBuffer`](crate::CommandBuffer), so the flush, which only
//! translates and submits what's recorded, is tested on a real device.

#[cfg(test)]
mod fake;
#[cfg(test)]
mod gpu_tests;
mod recording;
mod staging;

use std::sync::Arc;

use recording::{Recording, TransferQueue};

use crate::{
    AccessSet, Barrier, Buffer, CommandList, GpuResources, GpuTimepoint, Queue, QueueKind,
    RhiResult, SubmitDesc, TextureCopy,
};

/// Fills buffers and textures from any thread, on the transfer queue
/// (design 0002, "Streaming").
///
/// Each streaming thread has its own. Uploads are recorded until
/// [`flush`](Self::flush) submits them, and their bytes are staged in upload
/// buffers the uploader keeps, apart from rendering's transient memory, and
/// reuses once the copies that read them complete.
#[derive(Debug)]
pub struct Uploader {
    /// What's been uploaded since the last flush, and the staging.
    recording: Recording,
    /// The queue the copies are on.
    transfer: Arc<dyn Queue>,
    /// The queue that copies texture regions the transfer queue can't, and
    /// acquires what's uploaded.
    graphics: Arc<dyn Queue>,
    /// The timepoint the last flush returned, which a flush with nothing
    /// uploaded returns again rather than submitting nothing for one.
    last: Option<GpuTimepoint>,
}

impl Uploader {
    /// An uploader labelled `label`, which names its lists and staging
    /// buffers, filling `resources` on the `transfer` queue, and on the
    /// `graphics` queue where the transfer queue can't copy a texture
    /// region. Rendering acquires what it uploads on the graphics queue.
    ///
    /// # Panics
    ///
    /// Panics if `graphics` isn't a graphics queue, since the acquire
    /// barriers it returns are recorded there. The transfer queue may be any
    /// queue, as the queue of the kind transfers share with is when they
    /// have none of their own.
    pub fn new(
        resources: Arc<dyn GpuResources>,
        transfer: Arc<dyn Queue>,
        graphics: Arc<dyn Queue>,
        label: &str,
    ) -> Self {
        assert_eq!(
            graphics.kind(),
            QueueKind::Graphics,
            "uploader \"{label}\" is given a {:?} queue as its graphics queue",
            graphics.kind(),
        );
        let transfer_queue = TransferQueue {
            id: transfer.id(),
            granularity: transfer.transfer_granularity(),
        };
        Self {
            recording: Recording::new(resources, transfer_queue, graphics.id(), label),
            transfer,
            graphics,
            last: None,
        }
    }

    /// Uploads `data` to `buffer` at `offset`, leaving those bytes ready
    /// for `then` once the upload's acquire barriers are recorded. The rest
    /// of the buffer keeps its owner and contents.
    ///
    /// Nothing may use the range until the upload completes: the caller
    /// makes sure no submitted work still uses it.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`](crate::RhiError) if a staging
    /// buffer can't be created, or an error if the timelines of earlier
    /// flushes can't be read, having uploaded nothing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, `data` is empty or doesn't fit in the
    /// buffer at `offset`, `then` isn't an access set a buffer can have, or
    /// the buffer was uploaded to in this flush for other accesses.
    /// Overlapping bytes panic at the [`flush`](Self::flush).
    pub fn upload_buffer(
        &mut self,
        buffer: Buffer,
        offset: u64,
        data: &[u8],
        then: impl Into<AccessSet>,
    ) -> RhiResult<()> {
        self.reclaim()?;
        self.recording
            .upload_buffer(buffer, offset, data, then.into())
    }

    /// Uploads `data`, tightly packed texel blocks, to the texture region
    /// `copy` names, leaving its mip and layers ready for `then` once the
    /// upload's acquire barriers are recorded.
    ///
    /// The upload discards what the mip and layers held before this flush,
    /// so the flush must fill them whole, in one region or several.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`](crate::RhiError) if a staging
    /// buffer can't be created, or an error if the timelines of earlier
    /// flushes can't be read, having uploaded nothing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the texture can't be copied to, the
    /// region isn't in its mip, `data` isn't as long as the region's
    /// tightly packed blocks, `then` isn't an access set a texture can have,
    /// or the region's layers overlap others uploaded to in its mip in this
    /// flush without being the same, or are the same but for other accesses
    /// or on another queue. Overlapping regions, and mips not filled whole,
    /// panic at the [`flush`](Self::flush), and the rest of the copy is
    /// checked when the flush translates it.
    pub fn upload_texture(
        &mut self,
        copy: TextureCopy,
        data: &[u8],
        then: impl Into<AccessSet>,
    ) -> RhiResult<()> {
        self.reclaim()?;
        self.recording.upload_texture(copy, data, then.into())
    }

    /// Submits everything uploaded since the last flush, and returns when
    /// it completes and what rendering records before its first use.
    ///
    /// The transfer queue's copies are submitted first, then any the
    /// graphics queue makes, waiting for them, so the upload's timepoint is
    /// the last submission's, reached once every copy is complete. A flush
    /// with nothing uploaded submits nothing, and returns the last flush's
    /// timepoint, which is reached already or soon, or for the first flush,
    /// the timepoint of an empty submission to the transfer queue.
    ///
    /// Each flush first reuses the staging buffers of earlier flushes whose
    /// copies are complete, so that an uploader that only flushes, with
    /// nothing to upload, still lets its staging shrink.
    ///
    /// # Errors
    ///
    /// Returns an error if the timelines of earlier flushes can't be read,
    /// having submitted nothing and kept the uploads for the next flush, or
    /// if translation or submission fails. The uploads are lost then, and
    /// their staging buffers are destroyed once nothing submitted can use
    /// them. Both lists are translated before either is submitted, so a
    /// failed translation submits nothing. Only if submitting the graphics
    /// queue's list fails after the transfer queue's was submitted, which in
    /// practice means the device was lost, may the transfer queue's copies
    /// still run, and what they upload must be uploaded again before it's
    /// used.
    ///
    /// # Panics
    ///
    /// Panics if two uploads to a buffer in this flush overlap, two regions
    /// of a mip overlap, a mip uploaded to isn't filled whole, or a copy
    /// fails the checks translation makes (see
    /// [`CommandList::copy_buffer_to_texture`](crate::CommandList::copy_buffer_to_texture)).
    pub fn flush(&mut self) -> RhiResult<Upload> {
        self.reclaim_completed()?;
        let recorded = self.recording.finish();
        match self.submit(recorded.transfer, recorded.graphics) {
            Ok(timepoint) => {
                self.recording.staging().submitted(timepoint);
                Ok(Upload {
                    timepoint,
                    acquire: recorded.acquire,
                })
            }
            Err(err) => {
                self.recording.staging().discard();
                Err(err)
            }
        }
    }

    /// Translates the transfer queue's list and the graphics queue's, then
    /// submits the first, and the second waiting for it, and returns the
    /// last submission's timepoint. With neither list, it's the last flush's
    /// timepoint, or an empty submission's for the first flush.
    fn submit(
        &mut self,
        transfer: Option<CommandList>,
        graphics: Option<CommandList>,
    ) -> RhiResult<GpuTimepoint> {
        // Translating both first means a failed translation submits nothing.
        let transfer = transfer
            .map(|list| self.transfer.translate(&list))
            .transpose()?;
        let graphics = graphics
            .map(|list| self.graphics.translate(&list))
            .transpose()?;
        let mut last = None;
        if let Some(command_buffer) = transfer {
            last = Some(
                self.transfer
                    .submit(SubmitDesc::new(vec![command_buffer]))?,
            );
        }
        if let Some(command_buffer) = graphics {
            last = Some(self.graphics.submit(SubmitDesc {
                command_buffers: vec![command_buffer],
                wait_for: last.as_slice(),
            })?);
        }
        let timepoint = match last.or(self.last) {
            Some(timepoint) => timepoint,
            None => self.transfer.submit(SubmitDesc::default())?,
        };
        self.last = Some(timepoint);
        Ok(timepoint)
    }

    /// Reuses the staging buffers of earlier flushes whose copies are
    /// complete, before the first upload after a flush.
    fn reclaim(&mut self) -> RhiResult<()> {
        if !self.recording.staging().batch_is_empty() {
            return Ok(());
        }
        self.reclaim_completed()
    }

    /// Reuses the staging buffers of earlier flushes whose copies are
    /// complete.
    fn reclaim_completed(&mut self) -> RhiResult<()> {
        let (transfer, graphics) = (&self.transfer, &self.graphics);
        self.recording.staging().reclaim(|timepoint| {
            if timepoint.queue() == transfer.id() {
                transfer.is_complete(timepoint)
            } else {
                graphics.is_complete(timepoint)
            }
        })
    }
}

/// A finished upload: when its copies complete, and what rendering records
/// before it first uses what was uploaded.
///
/// Rendering waits for [`timepoint`](Self::timepoint), by passing it in a
/// [`SubmitDesc`]'s `wait_for`, and records the
/// [`acquire`](Self::acquire) barriers on the graphics queue before its first
/// use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upload {
    /// The timepoint every copy is complete at.
    pub timepoint: GpuTimepoint,
    /// The barriers rendering records on the graphics queue before it first
    /// uses what was uploaded, which acquire it from the transfer queue.
    /// Empty when the transfer queue is the graphics queue.
    pub acquire: Vec<Barrier>,
}

// An uploader moves to the thread that streams with it (design 0002,
// "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send`.
    const fn assert_send<T: Send>() {}
    assert_send::<Uploader>();
    assert_send::<Upload>();
};
