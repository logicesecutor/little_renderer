# webgpu/include

`webgpu/webgpu.h` is the standard WebGPU C header of the webgpu-native/webgpu-headers project
(Copyright 2019-2023 WebGPU-Native developers, BSD-3-Clause, the licence in the file), the
version wgpu-native v29.0.1.1 shipped. It is the only file of this folder and the only one of
the native build not written here: our Vulkan layer (`webgpu/vk`) implements it and the engine
(`engine/gpu`) is written against it. The browser build does not use it: Emscripten's
`--use-port=emdawnwebgpu` brings Dawn's copy of the same header.

What our layer adds to it (waiting for a submission, a second queue, ray tracing) is declared
in `webgpu/vk/webgpu_lr.h` and `webgpu/vk/webgpu_rt.h`, with names of our own (`lr...`, `LR...`).
