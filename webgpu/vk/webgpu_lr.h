// webgpu_lr.h - what our Vulkan layer adds to webgpu.h, apart from ray tracing (webgpu_rt.h):
// waiting for the GPU, and a second queue. Names of our own (lr..., LR...): webgpu.h has no
// way to wait for a submission (a browser cannot block), so every native implementation adds
// one (wgpu-native: wgpuDevicePoll, Dawn: wgpuInstanceWaitAny with futures); ours is this.
#ifndef LR_WEBGPU_LR_H
#define LR_WEBGPU_LR_H

#include <webgpu/webgpu.h>

// ---- waiting for the GPU ----
// A submission's number: lrQueueSubmitForIndex returns it, lrDevicePoll waits for it.
typedef uint64_t LRSubmissionIndex;

// wgpuQueueSubmit that also returns the submission's number (its queue in the top bits).
LRSubmissionIndex lrQueueSubmitForIndex(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const *commands);

// Gives back what finished submissions held and runs the callbacks that are due. With wait:
// first waits until the submission `submissionIndex` points to is done, or (NULL) everything
// submitted so far. Returns 1 when nothing submitted is still running.
WGPUBool lrDevicePoll(WGPUDevice device, WGPUBool wait, LRSubmissionIndex const *submissionIndex);

// ---- features of our own, for WGPUDeviceDescriptor.requiredFeatures ----
// (in a range of ours, like the SType values below: 0x4C52 is "LR")
#define LRFeatureName_RayQuery ((WGPUFeatureName)0x4C520100)   // webgpu_rt.h

// ---- a second queue ----
// To run compute (the physics) beside the main queue's work. Of a compute-only family when the
// GPU has one (its asynchronous compute engine), else of the main queue's family. Only when
// asked for: LR_VK_QUEUES=2 (or =same for a second of the main family); else this is the main
// queue itself. (On an RTX 3070 mesh_rain was slower with it: 35 frames/s against 45, the same
// family 42.)
// - Its command buffers come from encoders made for it: an LRCommandEncoderQueue chained
//   to the WGPUCommandEncoderDescriptor. Compute, copies and builds only (no render passes),
//   and buffers only (textures are the main queue's).
// - Nothing orders the two queues' work: a buffer one queue writes and the other uses needs
//   the first's submission waited for (lrDevicePoll with its lrQueueSubmitForIndex number)
//   before the second's submit.
WGPUQueue lrDeviceGetComputeQueue(WGPUDevice device);
#define LRSType_CommandEncoderQueue ((WGPUSType)0x4C520003)
typedef struct { WGPUChainedStruct chain; WGPUQueue queue; } LRCommandEncoderQueue;

#endif
