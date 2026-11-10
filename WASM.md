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
# build.jl: julia --project=<env> build.jl, in an environment with WasmCodegen, WasmCompat, SIMD and this FFTL,
# and FFTL's preference lazy_radices = [] (see below)
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

This example, as written, builds in about 3 minutes (wall time, including loading the packages and `wasm-opt -O3` on
two threads) into a 1.8 MB module, and prints 5, as native Julia does.

What each piece does:
- **`FFTLWasmCodegenExt`**, this package's extension, is loaded with WasmCodegen
  ([`ext/FFTLWasmCodegenExt.jl`](ext/FFTLWasmCodegenExt.jl)). Its methods are overlays that apply only when
  WasmCodegen compiles a program:
  - **The plans.** `plan_fft`, `plan_bfft`, `plan_ifft`, `plan_rfft`, `plan_brfft`, `plan_irfft` and `plan_r2r`
    make FFTL's run-time plans (`RPlan`, `RRFFTPlan`, `RR2RPlan`), and so do the functions built on them (`fft`,
    `rfft`, `dct`, `plan_dct`, ...). The normal plans carry their radix sequence in their type, so they cannot be
    compiled ahead of time. The run-time plans are the ones `juliac --trim` and libfftl use, with the same
    codelets.
  - **Threads.** Plans get one thread, and the helpers that start tasks run their items in order. FFTL's task
    helpers contain `Threads.@spawn`, and WasmCodegen compiles programs of one task by default.
  - **N-D DCT scaling.** `dct`, `idct` and `plan_dct(x) * x` of 2- to 6-dimensional arrays scale the first element
    of each dimension by index rather than with `selectdim`. On an N-D array, `selectdim` with a run-time dimension
    is a `SubArray` of unknown dimensionality, a dynamic call. Vectors use FFTL's own methods.
  - **Build hosts with target intrinsics.** On Apple silicon (FCMA) and x86 with AVX2 or AVX-512, the generic
    complex arithmetic replaces the LLVM target intrinsics.
- **FFTL's preference `lazy_radices = []`**, in the build environment's `LocalPreferences.toml`
  ([`wasm/LocalPreferences.toml`](wasm/LocalPreferences.toml)). By default FFTL compiles the codelets of radices
  17–31 at their first call; a module compiled ahead of time needs them all compiled with the package, as libfftl
  does.
- **WasmCompat's `WasmCompatSIMDExt`** compiles the arithmetic (`+`, `-`, `*`, `/`, `muladd`, ...), loads and
  stores of SIMD.jl's 16-byte float vectors (`Vec{2,Float64}`, `Vec{4,Float32}`, and `Vec{2,Float32}`) to WasmCodegen's
  SIMD128 (`v128`) operations, and runs the other vector intrinsics (shuffles among them) lane by lane.
  **`WasmCompatBaseExt`** makes `Base.generating_output()` false.
- **`relaxed_simd`** (WasmCodegen's `Config`, off by default): on, a vector `muladd` is relaxed SIMD's
  `relaxed_madd`, fused where the engine's hardware has FMA; off, a multiply and an add. `wasm/build.jl` takes
  `RELAXED_SIMD=1`.

Nothing else changes. FFTL doesn't time plans (the `flags` keyword is ignored natively too), its preferences are read
when it is precompiled, and the KernelAbstractions paths are not reached from CPU arrays.

[`wasm/build.jl`](wasm/build.jl) builds the release module the same way, and [`wasm/Project.toml`](wasm/Project.toml)
pins the versions it was built with.

## Limits

Measured on an aarch64 Linux host (Neoverse-N1) with Node 24.22.1 (V8), one thread. The module is `-O3`; native is
FFTL's normal plans with one thread. Each time is per transform with a precomputed plan (`mul!(y, p, x)`), the median
of 3 trials, then the median of 5 rounds:

| Transform | Size | Float64 native | Float64 Wasm | × | Float32 native | Float32 Wasm | × |
|---|---|---:|---:|---:|---:|---:|---:|
| `fft` | 64 | 0.261 µs | 1.7 µs | 6.5 | 0.168 µs | 1.48 µs | 8.8 |
| `fft` | 1024 | 7.32 µs | 44.2 µs | 6.0 | 4.47 µs | 34.6 µs | 7.7 |
| `fft` | 65536 | 904 µs | 4.94 ms | 5.5 | 543 µs | 4.13 ms | 7.6 |
| `fft` | 1009 | 45.6 µs | 234 µs | 5.1 | 31.3 µs | 209 µs | 6.7 |
| `ifft` | 1024 | 7.74 µs | 46.1 µs | 5.9 | 4.67 µs | 36.8 µs | 7.9 |
| `fft2` | 64×64 | 31.7 µs | 179 µs | 5.6 | 18.9 µs | 141 µs | 7.4 |
| `fft2` | 256×256 | 1.18 ms | 4.12 ms | 3.5 | 736 µs | 3.42 ms | 4.6 |
| `rfft` | 1024 | 4.12 µs | 24 µs | 5.8 | 2.89 µs | 20.7 µs | 7.2 |
| `rfft` | 65536 | 379 µs | 2.2 ms | 5.8 | 306 µs | 2.2 ms | 7.2 |
| `rfft` | 1009 | 46.3 µs | 265 µs | 5.7 | 31.6 µs | 263 µs | 8.3 |
| `irfft` | 1024 | 4.14 µs | 24.9 µs | 6.0 | 2.84 µs | 21.3 µs | 7.5 |
| `dct` | 1024 | 10 µs | 45.6 µs | 4.5 | 4.87 µs | 29.8 µs | 6.1 |
| `dct` | 1009 | 52.6 µs | 282 µs | 5.4 | 34.5 µs | 280 µs | 8.1 |

- **Speed:** 3.5–8.8× below native on these sizes (median 6.1×): 3.5–6.5× in Float64, 4.6–8.8× in Float32. That is
  9–24× faster than the previous release (`wasm-v2026-11-07`, 68–144× below native), from WasmCodegen's typed
  pointer loads and stores, its typed views of arrays, and SIMD128. Float32 gains less than Float64 in Wasm (it is
  1.0–1.5× faster than Float64 here, against 1.2–2.1× natively). What remains:
  - every access through a pointer computes an address, its offset in the array behind it, and range checks;
  - Wasm GC arrays have no 16-byte loads: a vector is loaded with two or four scalar `array.get`s;
  - SIMD.jl's shuffles run lane by lane.
  The Wasm times vary by at most 11% between rounds; the native times of the largest sizes (65536, 256×256) vary up
  to 3.7× between rounds on this shared host, so their ratios are the least precise.
- **Accuracy:** relative differences from native FFTL are at most 3.8e-16 (Float64) and 2.2e-7 (Float32), the same
  size as FFTW's distance from native FFTL on the same cases. Native fuses multiply-adds; this module does not
  (`relaxed_simd` off). Built with `RELAXED_SIMD=1`, 37 of the 50 cases are bitwise equal to native on this host
  (all Float32 ones), against 4 here, at the same speed; such a module needs an engine with relaxed SIMD, and its
  results depend on the machine.
- **Size:** the release module is 6.5 MB (6,504,099 bytes): all 18 transforms and the 12 bench exports; 12.4 MB before
  `wasm-opt`. It contains every codelet of the run-time chains (20 radices, both signs, the vector widths and the
  stage kinds) for both element types.
- **Build time:** about 8.5 minutes for the release module (wall time, on two threads): 2 minutes to compile and 5.5
  minutes of `wasm-opt -O3`, 1.5 GB peak.
- **Not covered:** threads, GPU arrays, element types other than Float32/Float64 in Wasm (FFTL's generic path is
  not tested there), and in-place real plans.

## What is still missing

| Missing | Where | Effect today |
|---|---|---|
| 16-byte loads from Wasm GC arrays, and `v128` storage | WasmCodegen | a vector load is a range check and two or four scalar `array.get`s |
| Cheaper pointer accesses | WasmCodegen | each access computes an address, an offset in its array and checks: most of the remaining gap |
| SIMD128 shuffles from SIMD.jl | WasmCompat | `shufflevector` runs lane by lane |
| `selectdim` with a run-time dimension on N-D arrays | WasmCodegen or FFTL | the N-D `dct`/`idct` scaling overlay in the extension |
| Tasks in the default JavaScript host | WasmCodegen | one thread (`Config(tasks = :jspi)` needs `host = :wasi` with `:julia_io` and an engine with JSPI) |
| Normal (compile-time) plans in Wasm | FFTL's design | the extension uses the run-time plans; a per-stage dispatch costs a little speed natively, not measured in Wasm |

## Follow-ups

- If FFTL's `dct`, `idct` and `OrthoDCTPlan` scaled the first element of each dimension by index (type-stable natively
  too), the N-D DCT overlay could go.
- FFTL's `RPlan{T}(sz, region::Integer, σ)` throws natively (`collect(Int, 2)` is 0-dimensional); the extension
  normalizes `region` itself. A one-line fix upstream would let it pass `region` through.
