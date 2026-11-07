# FFTL in WebAssembly

FFTL compiles to WebAssembly (Wasm GC) with [WasmCodegen.jl](https://github.com/gbaraldi/WasmCodegen.jl), a compiler
of Julia programs to Wasm. Unmodified FFTL code (`fft`, `ifft`, `bfft`, `rfft`, `irfft`, `dct`, `idct`, `plan_fft`
and `p * x`, `mul!`, ...) then runs in a browser, Node or Bun, in Float64 and Float32, for any length and any number
of dimensions. The results match native FFTL to rounding.

There are two ways to use it:
- **From JavaScript:** load the prebuilt module from the release and call its exports (*From JavaScript*).
- **From Julia:** compile a program of your own with WasmCodegen and this package's extension (*From Julia*).

## From JavaScript: the prebuilt module

The release `wasm-v<date>` has these files:
- `fftl.wasm`, optimized with `wasm-opt -O3`;
- `julia.mjs`, WasmCodegen's JavaScript host;
- `check.mjs` and `reference.json`, the check against native FFTL.

The module exports these transforms, each in Float64 (`_f64`) and Float32 (`_f32`). Their layouts are in the table
in [`wasm/fft.jl`](wasm/fft.jl).

| Export | Input | Output | FFTL call |
|---|---|---|---|
| `fft(n)`, `bfft(n)`, `ifft(n)` | n complex | n complex | `fft(x)`, `bfft(x)`, `ifft(x)` |
| `fft2(m, n)`, `ifft2(m, n)` | m×n complex | m×n complex | `fft(X)`, `ifft(X)` |
| `rfft(n)` | n real | n ÷ 2 + 1 complex | `rfft(x)` |
| `irfft(n)` | n ÷ 2 + 1 complex | n real | `irfft(X, n)` |
| `dct(n)`, `idct(n)` | n real | n real | `dct(x)` (orthonormal DCT-II), `idct(x)` |

A host passes numbers only:
1. it fills the input buffer (`input_resize`, `input_set`);
2. it calls a transform;
3. it reads the output buffer (`output_length`, `output_get`).

Arrays are column-major, and a complex number is two numbers, real then imaginary. The buffers hold Float64s; the
`_f32` transforms compute in Float32. Integers are 64-bit, so they are `BigInt`s in JavaScript.

```js
// node example.mjs  (Node 24; with fftl.wasm and julia.mjs from the release next to it)
import fs from 'node:fs';
import { instantiate } from './julia.mjs';
const { instance } = await instantiate(fs.readFileSync('fftl.wasm'));
const e = instance.exports;

// x = [1, 2, 3, 4] as complex numbers (real, imaginary, ...): X = fft(x)
const x = [1, 0, 2, 0, 3, 0, 4, 0];
e.input_resize(BigInt(x.length));
x.forEach((v, i) => e.input_set(BigInt(i + 1), v));
const status = e.fft_f64(4n);                 // 0: done; -1: the input's length does not match n
const X = Array.from({ length: Number(e.output_length()) }, (_, i) => e.output_get(BigInt(i + 1)));
console.log(status, X);                        // 0n [ 10, 0, -2, 2, -2, 0, -2, -2 ]
```

`node check.mjs` (or `bun check.mjs`) runs the release's 50 reference cases and compares the results. The inputs and
results come from native FFTL ([`wasm/reference.jl`](wasm/reference.jl)). A case is EXACT when the results are
bitwise equal, and CLOSE when the relative difference is within 1e-12 (Float64) or 1e-5 (Float32). Each case also
shows FFTW's distance from native FFTL, for scale.

The `bench_<op>_<f64|f32>(dims..., reps)` exports plan once and apply the plan `reps` times to a fixed input. They are
for timing.

## From Julia: compile your own program

Your program calls FFTL as usual. The build script loads WasmCodegen, FFTL, WasmCompat and SIMD:

```julia
# build.jl: julia --project=<env> build.jl, in an environment with WasmCodegen, WasmCompat (branch simd-lanes),
# SIMD and this FFTL
using WasmCodegen, FFTL, WasmCompat, SIMD

function peak(n::Int)                       # your program: plain FFTL
    x = [cos(2π * 5t / n) + 0.5sin(2π * 13t / n) for t in 0:n-1]
    X = rfft(x)
    return argmax(abs.(X)) - 1              # the strongest frequency bin: 5
end

c = WasmCodegen.compile([(peak, (Int,))])   # trim = :safe; optimized with wasm-opt -O3 (WasmCodegen's default)
write("peak.wasm", c.bytes)
cp(WasmCodegen.JS_HOST, "julia.mjs"; force = true)
```

```js
const { instance } = await instantiate(fs.readFileSync('peak.wasm'));
console.log(instance.exports.peak(1000n));   // 5n
```

This example, as written, builds in about 6 minutes (wall time, including precompilation and `wasm-opt -O3` on two
threads) into a 3.0 MB module, and prints 5, as native Julia does.

What each piece does:
- **`FFTLWasmCodegenExt`**, this package's extension, is loaded with WasmCodegen
  ([`ext/FFTLWasmCodegenExt.jl`](ext/FFTLWasmCodegenExt.jl)). Its methods are overlays that apply only when
  WasmCodegen compiles a program:
  - **The plans.** `plan_fft`, `plan_bfft`, `plan_ifft`, `plan_rfft`, `plan_brfft`, `plan_irfft` and `plan_r2r`
    make FFTL's run-time plans (`RPlan`, `RRFFTPlan`, `RR2RPlan`), and so do the functions built on them (`fft`,
    `rfft`, `dct`, `plan_dct`, ...). The normal plans carry their radix sequence in their type, so they cannot be
    compiled ahead of time. The run-time plans are the ones `juliac --trim` and libfftl use, with the same
    codelets.
  - **Threads.** Plans get one thread, and the helpers that start tasks run their items in order.
  - **DCT scaling.** `dct`, `idct` and `plan_dct` scale the first element of each dimension by index rather than
    with `selectdim`. With a run-time dimension, `selectdim` made an invalid module before
    [WasmCodegen.jl#80](https://github.com/gbaraldi/WasmCodegen.jl/pull/80).
  - **Build hosts with target intrinsics.** On Apple silicon (FCMA) and x86 with AVX2, the generic complex
    arithmetic replaces the LLVM target intrinsics.
- **WasmCompat's `WasmCompatSIMDExt`** runs SIMD.jl's vector intrinsics (the codelets' arithmetic and shuffles) lane
  by lane. **`WasmCompatBaseExt`** makes `Base.generating_output()` false. Both are on the `simd-lanes` branch,
  [WasmCompat.jl#6](https://github.com/pankgeorg/WasmCompat.jl/pull/6).

Nothing else changes. FFTL doesn't time plans (the `flags` keyword is ignored natively too), its preferences are read
when it is precompiled, and the KernelAbstractions paths are not reached from CPU arrays.

[`wasm/build.jl`](wasm/build.jl) builds the release module the same way, and [`wasm/Project.toml`](wasm/Project.toml)
pins the versions it was built with.

## Limits

Measured on an aarch64 Linux host with Node 24.22.1 (V8), one thread. The modules are `-O3`; native is FFTL's normal
plans with one thread. Each time is per transform with a precomputed plan (`mul!(y, p, x)`), the median of 3 trials:

| Transform | Size | Float64 native | Float64 Wasm | × | Float32 native | Float32 Wasm | × |
|---|---|---:|---:|---:|---:|---:|---:|
| `fft` | 64 | 0.26 µs | 38.6 µs | 149 | 0.168 µs | 22.5 µs | 134 |
| `fft` | 1024 | 7.25 µs | 918 µs | 127 | 4.42 µs | 595 µs | 135 |
| `fft` | 65536 | 904 µs | 111 ms | 123 | 522 µs | 67 ms | 128 |
| `fft` | 1009 | 45.4 µs | 3.92 ms | 86 | 31 µs | 2.25 ms | 72 |
| `ifft` | 1024 | 7.67 µs | 922 µs | 120 | 4.63 µs | 569 µs | 123 |
| `fft2` | 64×64 | 31.8 µs | 3.94 ms | 124 | 18.9 µs | 2.53 ms | 133 |
| `fft2` | 256×256 | 1.11 ms | 91.3 ms | 83 | 500 µs | 53.3 ms | 107 |
| `rfft` | 1024 | 4.12 µs | 472 µs | 115 | 2.85 µs | 288 µs | 101 |
| `rfft` | 65536 | 893 µs | 56.3 ms | 63 | 314 µs | 34.6 ms | 110 |
| `rfft` | 1009 | 46.1 µs | 4.22 ms | 91 | 31.4 µs | 2.41 ms | 77 |
| `irfft` | 1024 | 4.09 µs | 474 µs | 116 | 2.81 µs | 287 µs | 102 |
| `dct` | 1024 | 8.52 µs | 678 µs | 80 | 4.82 µs | 403 µs | 84 |
| `dct` | 1009 | 52.1 µs | 4.34 ms | 83 | 36.7 µs | 2.48 ms | 68 |

- **Speed:** about 100× below native: 63–149× on these sizes, the lower end (63–91×) on prime lengths (Bluestein),
  on `dct` and on some large sizes. Three things account for it:
  - Wasm here has no SIMD: SIMD.jl's 16-byte vectors run lane by lane. WasmCodegen has no SIMD128 lowering yet.
  - FFTL's codelets load and store through pointers, which WasmCodegen reads and writes one byte at a time without
    a static origin.
  - There are no FMA instructions.
- **Accuracy:** relative differences from native FFTL are at most 3.8e-16 (Float64) and 2.2e-7 (Float32), the same
  size as FFTW's distance from native FFTL on the same cases. Native uses fused multiply-adds and the compile-time
  chains; Wasm uses neither.
- **Size:** the release module is 9.8 MB (9,830,248 bytes): all 18 transforms and the 12 bench exports; 14.1 MB before
  `wasm-opt`. It contains every codelet of the run-time chains (20 radices, both signs, the vector widths and the
  stage kinds) for both element types.
- **Build time:** the release module took about 2 minutes to compile and 42 minutes of `wasm-opt -O3` on two
  threads (about 5 GB peak). `wasm-opt` dominates.
- **Not covered:** threads, GPU arrays, element types other than Float32/Float64 in Wasm (FFTL's generic path is
  not tested there), and in-place real plans.

## What is still missing

| Missing | Where | Effect today |
|---|---|---|
| SIMD128 lowering of SIMD.jl's 16-byte vectors | WasmCodegen | lanes run one by one (`WasmCompatSIMDExt`) |
| Typed 4/8-byte loads through pointers without a static origin | WasmCodegen | byte-wise loads and stores in every codelet |
| Smaller input to `wasm-opt` (reuse of locals) | WasmCodegen | 42 min of `wasm-opt` for the release module |
| `selectdim` with a run-time dimension ([#80](https://github.com/gbaraldi/WasmCodegen.jl/pull/80)) | WasmCodegen | `dct`/`idct` scaling overlay in the extension |
| Normal (compile-time) plans in Wasm | FFTL's design | the extension uses the run-time plans; a per-stage dispatch costs a little speed natively, not measured in Wasm |
| SIMD.jl lanes merged into WasmCompat main | WasmCompat [#6](https://github.com/pankgeorg/WasmCompat.jl/pull/6) | `wasm/Project.toml` pins the `simd-lanes` branch |

## Follow-ups

- When [WasmCodegen.jl#80](https://github.com/gbaraldi/WasmCodegen.jl/pull/80) is merged, the DCT-scaling overlay
  in the extension (`dct`, `idct`, `*(::OrthoDCTPlan, x)`, ~30 lines) can go for one-dimensional transforms. On N-D
  arrays, `selectdim` with a run-time dimension is a dynamic call (a `SubArray` of unknown dimensionality), so N-D
  `dct` still needs the overlay or an FFTL change that scales by index.
- When [WasmCompat.jl#6](https://github.com/pankgeorg/WasmCompat.jl/pull/6) is merged, `wasm/Project.toml` can take
  WasmCompat from main instead of the `simd-lanes` branch.
