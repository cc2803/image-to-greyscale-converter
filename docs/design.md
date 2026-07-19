# High-Level Design: GPU-Accelerated Image Grayscale Conversion

## 1. Overview

This document describes the high-level design of a CUDA-based pipeline that
converts an RGB image to grayscale. The workload is offloaded from the CPU to
the GPU, where each pixel is processed independently by a dedicated thread.

The problem is classified as **embarrassingly parallel**: no pixel's
computation depends on any other pixel's result. This property makes it a
canonical introductory workload for GPU programming and a useful baseline for
reasoning about memory-bound CUDA kernels.

## 2. Goals and Non-Goals

**Goals**
- Convert an RGB image to grayscale correctly, using a perceptually weighted
  luminance formula.
- Maximize throughput by exploiting per-pixel data parallelism on the GPU.
- Keep host↔device data transfer minimal, since it is the dominant cost.

**Non-Goals**
- Optimizing the arithmetic itself (the per-pixel computation is already
  trivial — three multiplies and two adds).
- Support for formats beyond flat RGB/RGBA pixel buffers (e.g., no color
  space or codec-specific handling is covered here).
- Multi-GPU or distributed execution.

## 3. System Context

```
+--------------------+                         +----------------------+
|        Host (CPU)  |                         |      Device (GPU)    |
+--------------------+                         +----------------------+
| Decode image (PNG)  |                        |                      |
| via stb_image/OpenCV|                        |                      |
+---------+-----------+                        +----------+-----------+
          |                                                ^
          | RGB pixel buffer                               |
          | cudaMemcpy (Host -> Device)                     |
          v                                                |
+----------------------+                +-------------------+---------+
| GPU Global Memory     |  ------------> |   CUDA Kernel               |
| Input RGB Buffer      |                |   1 thread : 1 pixel        |
+----------------------+                |   RGB -> Grayscale          |
                                          +-------------------+---------+
+----------------------+                                    |
| GPU Global Memory     |  <-------------------------------- |
| Output Gray Buffer    |
+----------------------+
          |
          | cudaMemcpy (Device -> Host)
          v
+----------------------+
| Encode grayscale PNG |
+----------------------+
```

## 4. Pipeline Stages

| Stage | Component | Description |
|---|---|---|
| 1 | Image Load (Host) | Decode PNG/JPEG into a flat `unsigned char*` RGB buffer using stb_image or OpenCV. |
| 2 | Device Allocation | `cudaMalloc` input and output buffers on the GPU. |
| 3 | Host → Device Transfer | `cudaMemcpy` the RGB buffer across the PCIe bus into GPU global memory. |
| 4 | Kernel Launch | Launch a grid of thread blocks; each thread maps to exactly one pixel. |
| 5 | Per-Pixel Compute | Each thread reads R, G, B and computes the grayscale value. |
| 6 | Output Write | Each thread writes 1 byte (grayscale) or 3 bytes (replicated R=G=B) to the output buffer. |
| 7 | Device → Host Transfer | `cudaMemcpy` the result buffer back to host memory. |
| 8 | Image Save (Host) | Encode the grayscale buffer to PNG/JPEG and write to disk. |

### 4.1 Data Layout

Input pixels are stored as an interleaved byte stream:

```
Pixel 0    Pixel 1    Pixel 2
R  G  B    R  G  B    R  G  B  ...
```

For a `1920 x 1080` image:

- Pixel count: `1920 x 1080 = 2,073,600`
- Input buffer size: `2,073,600 x 3 bytes ≈ 6.2 MB`
- Output buffer size (1 byte/pixel): `2,073,600 bytes ≈ 2.1 MB`

### 4.2 Thread-to-Pixel Mapping

CUDA schedules work in a grid of blocks, each containing a fixed number of
threads (commonly 256). Each thread computes its own global pixel index:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

For `2,000,000` pixels at `blockDim = 256`:

```
grid size = ceil(2,000,000 / 256) ≈ 7,813 blocks
```

### 4.3 Grayscale Computation

Each thread applies the ITU-R BT.601 perceptual luminance weighting:

```cuda
gray = 0.299f * R + 0.587f * G + 0.114f * B;
```

Green is weighted most heavily because human vision is most sensitive to
green light, followed by red, then blue.

## 5. Key Design Decisions

| Decision | Rationale |
|---|---|
| One thread per pixel | Pixels are independent; this maximizes parallelism with no synchronization overhead. |
| Global memory only, no shared memory | Each pixel is read exactly once and written exactly once — shared memory offers no reuse benefit here. |
| Block size of 256 threads | Balances occupancy against register pressure; a standard default for memory-bound kernels. |
| Minimize host↔device transfers | Transfers over PCIe dominate total runtime far more than the arithmetic itself. |
| No branching in the kernel | Avoids warp divergence; all 32 threads in a warp follow the same instruction path. |

## 6. Memory Hierarchy Considerations

```
Registers        (fastest, per-thread)
Shared Memory     (per-block, unused in this design)
L1 Cache
Global Memory     (used for input/output buffers)
Host Memory       (slowest, reached only via PCIe)
```

Because every pixel is touched exactly once, this workload has no data reuse
to exploit — global memory access, done with proper coalescing, is
sufficient.

**Memory coalescing**: adjacent threads must access adjacent memory
addresses so the GPU can merge them into a single memory transaction.

```
Good:  Thread 0 -> byte 0,  Thread 1 -> byte 1,  Thread 2 -> byte 2 ...
Bad:   Thread 0 -> byte 0,  Thread 1 -> byte 100, Thread 2 -> byte 9000 ...
```

## 7. Performance Characteristics

This kernel is **memory-bound**, not compute-bound: for every pixel, the GPU
performs three reads, three multiplies, two adds, and one write — a
negligible amount of arithmetic relative to the memory traffic involved.

Bottleneck ranking (highest to lowest impact):

1. Host ↔ device PCIe transfer (both directions)
2. Global memory read/write bandwidth
3. Kernel launch overhead
4. Arithmetic (negligible)

**Warp execution**: CUDA schedules threads in warps of 32. Because the
kernel contains no conditional branching, all threads in a warp always
execute the same instruction path — warp divergence is effectively zero.

## 8. Trade-offs and Alternatives Considered

| Alternative | Considered | Reason Not Chosen |
|---|---|---|
| CPU-only (single/multi-threaded loop) | Yes | Serializes or under-parallelizes an embarrassingly parallel workload; far slower at scale. |
| Shared-memory tiling | Yes | No benefit — each pixel is read once, so there is no data reuse to cache. |
| Larger block sizes (512+) | Yes | Diminishing returns and increased register pressure without a proportional occupancy gain. |
| Output as 3-byte replicated grayscale (R=G=B) vs 1-byte | Yes | Kept as an implementation option; depends on downstream image-writer requirements. |

## 9. Future Extensions

- **Streaming / overlap**: use CUDA streams to overlap `cudaMemcpy` with
  kernel execution across image tiles or batches, hiding transfer latency.
- **Batch processing**: extend the kernel to process multiple images per
  launch to amortize kernel-launch overhead.
- **Unified memory**: evaluate `cudaMallocManaged` to simplify the memory
  model at a potential performance cost, useful for prototyping.
- **Additional filters**: the same 1-thread-per-pixel pattern generalizes to
  brightness/contrast adjustment, thresholding, and simple convolution
  filters (blur, edge detection).

## 10. Summary

The design's core insight is that GPU acceleration here isn't about
optimizing the grayscale formula — it's about restructuring a sequential
CPU loop into thousands of independent, concurrently executing threads,
while keeping memory access patterns coalesced and host↔device transfers to
a minimum. This pattern generalizes to a broad class of data-parallel
image-processing workloads.
