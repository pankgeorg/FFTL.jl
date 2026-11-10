"""
    FFTLWasmCodegenExt

FFTL in WebAssembly, compiled by [WasmCodegen](https://github.com/gbaraldi/WasmCodegen.jl). Loaded
with WasmCodegen; see `WASM.md`. Its methods are overlays in WasmCodegen's method table
(`WasmCodegen.Runtime.WASM_METHODS`): they replace FFTL's own only when WasmCodegen compiles a
program, so native FFTL is unchanged.

- **Plans whose kernels are data.** The normal plans (`FFTLPlan`, `RFFTLPlan`, `R2RPlan`) carry
  their radix sequence in their type, which is computed from the size when the plan is made: a
  program compiled ahead of time cannot contain them. In Wasm, `plan_fft`, `plan_bfft`,
  `plan_ifft`, `plan_rfft`, `plan_brfft`, `plan_irfft` and `plan_r2r` (and so `fft`, `rfft`, `dct`,
  `plan_dct`, …) make FFTL's run-time plans instead (`RPlan`, `RRFFTPlan`, `RR2RPlan`: the plans of
  `juliac --trim` and libfftl), which compute the same transforms with the same codelets. The real
  and r2r run-time plans run on pointers; `RealPlan`, `BRealPlan` and `R2RArrayPlan` here apply them
  to arrays, as AbstractFFTs plans.
- **One thread.** Plans get one thread whatever `threads` or `FFTL.set_num_threads` says, and the
  helpers that start tasks run their items in order, so `Threads.@spawn` is never compiled.
- **N-D orthonormal DCT.** `dct`, `idct` and `plan_dct(x) * x` of 2- to 6-dimensional arrays scale by
  index instead of `selectdim` with a run-time dimension (a dynamic call). Vectors use FFTL's own
  methods.
- **Build hosts with target intrinsics.** On an AArch64 host with the complex-number extension
  (Apple silicon) and on an x86 host with AVX2, FFTL's complex arithmetic calls LLVM target
  intrinsics (`llvm.aarch64.neon.vcadd.*`, `vcmla.*`, `llvm.x86.*.addsub.*`); here they take
  FFTL's generic form, which the other hosts use natively.

Element types other than `Float32` and `Float64` (real or complex) keep FFTL's own methods. FFTL does
not time plans (the `flags` are ignored natively too), its preferences are read when it is
precompiled, and the KernelAbstractions paths are not reached from CPU arrays: none needs an
overlay. The SIMD.jl vector intrinsics of the codelets are generic: `WasmCompatSIMDExt` (WasmCompat.jl)
maps them to WasmCodegen's SIMD128 (`v128`) where it can and runs the rest lane by lane. A module
compiled ahead of time also needs FFTL's preference `lazy_radices = []` (see `wasm/LocalPreferences.toml`):
otherwise the codelets of radices 17–31 are compiled at their first call.
"""
module FFTLWasmCodegenExt

using FFTL, WasmCodegen, LinearAlgebra
using FFTL: FFTLFloat, RPlan, RRFFTPlan, RR2RPlan, R2RKind, execute_r2c!, execute_c2r!, execute_r2r!
using FFTL.SIMD: Vec
import AbstractFFTs
import Base.Experimental: @overlay
const MT = WasmCodegen.Runtime.WASM_METHODS

const RArray{T<:FFTLFloat} = AbstractArray{T}
const CArray{T<:FFTLFloat} = AbstractArray{Complex{T}}

## One thread
@overlay MT FFTL.plan_threads() = 1
@overlay MT FFTL.planthreads(threads, num_threads) = 1
@overlay MT function FFTL.spawn_items(f::F, nitems, ntasks) where {F}
  for it in 0:nitems-1; f(1, it); end
end
@overlay MT function FFTL.spawn_phases(ntasks, n1, f1::F1, n2, f2::F2) where {F1,F2}
  for it in 0:n1-1; f1(1, it); end
  for it in 0:n2-1; f2(1, it); end
end

## Plans whose kernels are data
"r2c plan of real `N`-dimensional arrays: FFTL's `RRFFTPlan`, applied out of place."
struct RealPlan{T<:FFTLFloat,N} <: AbstractFFTs.Plan{T}
  p::RRFFTPlan{T}
  sz::NTuple{N,Int}
  osz::NTuple{N,Int}
end
"Unnormalised c2r plan to real arrays of size `osz`: FFTL's `RRFFTPlan`, out of place (the input is kept)."
struct BRealPlan{T<:FFTLFloat,N} <: AbstractFFTs.Plan{Complex{T}}
  p::RRFFTPlan{T}
  sz::NTuple{N,Int}
  osz::NTuple{N,Int}
end
"Real-to-real plan: FFTL's `RR2RPlan`, applied out of place."
struct R2RArrayPlan{T<:FFTLFloat,N} <: AbstractFFTs.Plan{T}
  p::RR2RPlan{T}
  sz::NTuple{N,Int}
end

region1(region, N) = FFTL.normregion(region, N)
regvec(region) = region isa Integer ? [Int(region)] : collect(Int, region)
half(sz::NTuple{N,Int}, d) where {N} = ntuple(i -> i == d ? sz[i] ÷ 2 + 1 : sz[i], Val(N))
setdim(sz::NTuple{N,Int}, d, n) where {N} = ntuple(i -> i == d ? Int(n) : sz[i], Val(N))

cplan(x::AbstractArray, region, σ) = RPlan{FFTL.plan_type(x)}(collect(Int, size(x)), regvec(region), σ; threads = 1)
function iplan(x::AbstractArray, region)
  T = FFTL.plan_type(x)
  p = RPlan{T}(collect(Int, size(x)), regvec(region), 1; threads = 1)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(T, size(x), p.region))
end
function rplan(x::RArray{T}, region) where {T}
  reg = region1(region, ndims(x))
  RealPlan{T,ndims(x)}(RRFFTPlan{T}(collect(Int, size(x)), reg, -1; threads = 1), size(x), half(size(x), reg[1]))
end
function brplan(X::CArray{T}, d::Integer, region) where {T}
  reg = region1(region, ndims(X))
  osz = setdim(size(X), reg[1], d)
  BRealPlan{T,ndims(X)}(RRFFTPlan{T}(collect(Int, osz), reg, 1; threads = 1), size(X), osz)
end
function irplan(X::CArray{T}, d::Integer, region) where {T}
  p = brplan(X, d, region)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(T, p.osz, p.p.region))
end
function r2rplan(x::AbstractArray, kinds, region)
  T = FFTL.realtype(x)
  reg = regvec(region)
  ks = kinds isa R2RKind ? fill(kinds, length(reg)) : collect(R2RKind, kinds)
  R2RArrayPlan{T,ndims(x)}(RR2RPlan{T}(collect(Int, size(x)), ks, reg; threads = 1), size(x))
end

Base.size(q::Union{RealPlan,BRealPlan,R2RArrayPlan}) = q.sz
AbstractFFTs.fftdims(q::Union{RealPlan,BRealPlan,R2RArrayPlan}) = Tuple(q.p.region)
AbstractFFTs.output_size(q::Union{RealPlan,BRealPlan}) = q.osz

function LinearAlgebra.mul!(y::Array{Complex{T},N}, q::RealPlan{T,N}, x::Array{T,N}) where {T,N}
  (size(x) == q.sz && size(y) == q.osz) || throw(DimensionMismatch("plan size does not match the arrays"))
  GC.@preserve x y execute_r2c!(q.p, pointer(y), pointer(x))
  y
end
function LinearAlgebra.mul!(y::Array{T,N}, q::BRealPlan{T,N}, x::Array{Complex{T},N}) where {T,N}
  (size(x) == q.sz && size(y) == q.osz) || throw(DimensionMismatch("plan size does not match the arrays"))
  GC.@preserve x y execute_c2r!(q.p, pointer(y), pointer(x))
  y
end
function LinearAlgebra.mul!(y::Array{T,N}, q::R2RArrayPlan{T,N}, x::Array{T,N}) where {T,N}
  (size(x) == q.sz && size(y) == q.sz) || throw(DimensionMismatch("plan size does not match the arrays"))
  GC.@preserve x y execute_r2r!(q.p, pointer(y), pointer(x))
  y
end
Base.:*(q::RealPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N} =
  mul!(Array{Complex{T},N}(undef, q.osz), q, convert(Array{T,N}, x))
Base.:*(q::BRealPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N} =
  mul!(Array{T,N}(undef, q.osz), q, convert(Array{Complex{T},N}, x))
Base.:*(q::R2RArrayPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N} =
  mul!(Array{T,N}(undef, q.sz), q, convert(Array{T,N}, x))

const FArray = Union{RArray,CArray}
@overlay MT FFTL.plan_fft(x::FArray, region = 1:ndims(x); kw...) = cplan(x, region, -1)
@overlay MT FFTL.plan_bfft(x::FArray, region = 1:ndims(x); kw...) = cplan(x, region, 1)
@overlay MT FFTL.plan_ifft(x::FArray, region = 1:ndims(x); kw...) = iplan(x, region)
@overlay MT FFTL.plan_rfft(x::RArray, region = 1:ndims(x); kw...) = rplan(x, region)
@overlay MT FFTL.plan_brfft(X::CArray, d::Integer, region = 1:ndims(X); kw...) = brplan(X, d, region)
@overlay MT FFTL.plan_irfft(X::CArray, d::Integer, region = 1:ndims(X); kw...) = irplan(X, d, region)
@overlay MT FFTL.plan_r2r(x::RArray, kinds, region = 1:ndims(x); kw...) = r2rplan(x, kinds, region)

## The orthonormal DCT of N-D arrays without `selectdim`. FFTL scales the first element of each transformed
## dimension with `selectdim(y, d, 1)`; with a run-time `d` that is a `SubArray` of unknown dimensionality on an N-D
## array, a dynamic call WasmCodegen does not compile. Here the same scaling is done by index. Vectors keep FFTL's own
## methods (they compile since gbaraldi/WasmCodegen.jl#80).
const NDArray{T} = Union{AbstractArray{T,2},AbstractArray{T,3},AbstractArray{T,4},AbstractArray{T,5},AbstractArray{T,6}}
function scale_first!(y::AbstractArray, d::Int, s)
  S, outer = FFTL.dim_extent(size(y), d)
  n = size(y, d)
  for o in 0:outer-1, i in 1:S
    y[i + S * n * o] *= s
  end
  y
end
dims_of(region) = region isa Integer ? (Int(region),) : region
function ortho_dct2!(y::AbstractArray{T}, region) where {T}
  for d in dims_of(region)
    y .*= inv(sqrt(T(2size(y, d))))
    scale_first!(y, d, inv(sqrt(T(2))))
  end
  y
end
function ortho_dct3_scale!(y::AbstractArray{T}, region) where {T}
  for d in dims_of(region)
    scale_first!(y, d, sqrt(T(2)))
    y .*= inv(sqrt(T(2size(y, d))))
  end
  y
end
dct2(x::AbstractArray, region) = ortho_dct2!(FFTL.r2r(x, FFTL.REDFT10, region), region)
dct3(x::AbstractArray, region) =
  FFTL.r2r(ortho_dct3_scale!(copyto!(similar(x, FFTL.realtype(x)), x), region), FFTL.REDFT01, region)
ortho_mul(q::FFTL.OrthoDCTPlan{T}, x::AbstractArray) where {T} =
  q.inverse ? q.p * ortho_dct3_scale!(copyto!(similar(x, T), x), q.region) : ortho_dct2!(q.p * x, q.region)

@overlay MT FFTL.dct(x::NDArray{<:FFTLFloat}, region = 1:ndims(x)) = dct2(x, region)
@overlay MT FFTL.idct(x::NDArray{<:FFTLFloat}, region = 1:ndims(x)) = dct3(x, region)
@overlay MT Base.:*(q::FFTL.OrthoDCTPlan{T}, x::NDArray) where {T<:FFTLFloat} = ortho_mul(q, x)

## Build hosts with target intrinsics: FFTL's generic complex arithmetic (complexvec.jl)
@overlay MT @inline FFTL.fcadd90(a::Vec{N,T}, b::Vec{N,T}) where {N,T} =
  (sb = FFTL.swap_reim(b); FFTL.blend_reim(a - sb, a + sb))
@overlay MT @inline FFTL.fcadd270(a::Vec{N,T}, b::Vec{N,T}) where {N,T} =
  (sb = FFTL.swap_reim(b); FFTL.blend_reim(a + sb, a - sb))
@overlay MT @inline function FFTL.ccmul(a::Vec{N,T}, w::Vec{N,T}) where {N,T}
  t = FFTL.dup_im(a) * FFTL.swap_reim(w)
  FFTL.blend_reim(muladd(FFTL.dup_re(a), w, -t), muladd(FFTL.dup_re(a), w, t))
end

end # module
