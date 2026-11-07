# The entry points of the prebuilt module (wasm/build.jl): FFTL's common transforms in Float64 and Float32, through
# its normal API (in Wasm, FFTLWasmCodegenExt makes the plans). A host passes numbers only: it fills the input
# buffer, calls a transform, and reads the output buffer. Arrays are column-major; a complex number is two numbers,
# real then imaginary. The buffers hold Float64; the `_f32` transforms compute in Float32.
#
#   input_resize(len); input_set(i, x) for i in 1:len     the input array
#   <op>_<f64|f32>(dims...) -> status                       0: done; -1: the input's length does not match dims
#   output_length(); output_get(i)                          the result
#   bench_<op>_<f64|f32>(dims..., reps) -> 0                plans once, then applies the plan `reps` times to a
#                                                          fixed input (for timing; the buffers are not used)
#
# | Transform      | Input                  | Output                     | FFTL call               |
# |----------------|------------------------|----------------------------|-------------------------|
# | fft(n)         | n complex              | n complex                  | fft(x)                  |
# | bfft(n)        | n complex              | n complex                  | bfft(x) (unnormalised)  |
# | ifft(n)        | n complex              | n complex                  | ifft(x)                 |
# | fft2(m, n)     | m×n complex            | m×n complex                | fft(X)                  |
# | ifft2(m, n)    | m×n complex            | m×n complex                | ifft(X)                 |
# | rfft(n)        | n real                 | n ÷ 2 + 1 complex          | rfft(x)                 |
# | irfft(n)       | n ÷ 2 + 1 complex      | n real                     | irfft(X, n)             |
# | dct(n)         | n real                 | n real                     | dct(x) (orthonormal II) |
# | idct(n)        | n real                 | n real                     | idct(x) (DCT-III)       |
module FFTLWasm
using FFTL, LinearAlgebra

const IN = Float64[]
const OUT = Float64[]
input_resize(len::Int) = (resize!(IN, len); fill!(IN, 0.0); length(IN))
input_set(i::Int, x::Float64) = (@inbounds IN[i] = x; 0)
output_length() = length(OUT)
output_get(i::Int) = @inbounds OUT[i]

function cin(::Type{T}, dims::Int...) where {T}
  A = Array{Complex{T}}(undef, dims)
  @inbounds for i in eachindex(A)
    A[i] = Complex{T}(IN[2i-1], IN[2i])
  end
  A
end
function rin(::Type{T}, dims::Int...) where {T}
  A = Array{T}(undef, dims)
  @inbounds for i in eachindex(A)
    A[i] = T(IN[i])
  end
  A
end
function emit!(A::AbstractArray{<:Complex})
  empty!(OUT)
  for x in A
    push!(OUT, Float64(real(x))); push!(OUT, Float64(imag(x)))
  end
  0
end
function emit!(A::AbstractArray{<:Real})
  empty!(OUT)
  for x in A
    push!(OUT, Float64(x))
  end
  0
end

# fixed inputs for bench_*
cfix(::Type{T}, dims::Int...) where {T} = (A = Array{Complex{T}}(undef, dims); for i in eachindex(A); A[i] = Complex{T}(sin(0.37i), cos(0.11i)); end; A)
rfix(::Type{T}, dims::Int...) where {T} = (A = Array{T}(undef, dims); for i in eachindex(A); A[i] = T(sin(0.37i)); end; A)
function apply_reps(p, x, y, reps::Int)
  for _ in 1:reps
    mul!(y, p, x)
  end
  0
end

for (s, T) in (("f64", Float64), ("f32", Float32))
  C = Complex{T}
  @eval begin
    $(Symbol(:fft_, s))(n::Int) = length(IN) == 2n ? emit!(fft(cin($T, n))) : -1
    $(Symbol(:bfft_, s))(n::Int) = length(IN) == 2n ? emit!(bfft(cin($T, n))) : -1
    $(Symbol(:ifft_, s))(n::Int) = length(IN) == 2n ? emit!(ifft(cin($T, n))) : -1
    $(Symbol(:fft2_, s))(m::Int, n::Int) = length(IN) == 2m * n ? emit!(fft(cin($T, m, n))) : -1
    $(Symbol(:ifft2_, s))(m::Int, n::Int) = length(IN) == 2m * n ? emit!(ifft(cin($T, m, n))) : -1
    $(Symbol(:rfft_, s))(n::Int) = length(IN) == n ? emit!(rfft(rin($T, n))) : -1
    $(Symbol(:irfft_, s))(n::Int) = length(IN) == 2(n ÷ 2 + 1) ? emit!(irfft(cin($T, n ÷ 2 + 1), n)) : -1
    $(Symbol(:dct_, s))(n::Int) = length(IN) == n ? emit!(dct(rin($T, n))) : -1
    $(Symbol(:idct_, s))(n::Int) = length(IN) == n ? emit!(idct(rin($T, n))) : -1

    $(Symbol(:bench_fft_, s))(n::Int, reps::Int) = (x = cfix($T, n); apply_reps(plan_fft(x), x, similar(x), reps))
    $(Symbol(:bench_ifft_, s))(n::Int, reps::Int) = (x = cfix($T, n); apply_reps(plan_ifft(x), x, similar(x), reps))
    $(Symbol(:bench_fft2_, s))(m::Int, n::Int, reps::Int) = (x = cfix($T, m, n); apply_reps(plan_fft(x), x, similar(x), reps))
    $(Symbol(:bench_rfft_, s))(n::Int, reps::Int) =
      (x = rfix($T, n); apply_reps(plan_rfft(x), x, Vector{$C}(undef, n ÷ 2 + 1), reps))
    $(Symbol(:bench_irfft_, s))(n::Int, reps::Int) =
      (x = cfix($T, n ÷ 2 + 1); apply_reps(plan_brfft(x, n), x, Vector{$T}(undef, n), reps))
    $(Symbol(:bench_dct_, s))(n::Int, reps::Int) = (x = rfix($T, n); apply_reps(plan_dct(x), x, similar(x), reps))
  end
end

const OPS = (:fft => (Int,), :bfft => (Int,), :ifft => (Int,), :fft2 => (Int, Int), :ifft2 => (Int, Int),
             :rfft => (Int,), :irfft => (Int,), :dct => (Int,), :idct => (Int,))
const BENCH = (:fft => (Int, Int), :ifft => (Int, Int), :fft2 => (Int, Int, Int), :rfft => (Int, Int),
               :irfft => (Int, Int), :dct => (Int, Int))
"The module's exports: `(function, argument types)`, for `WasmCodegen.compile`."
function entries(; ops = first.(OPS), types = ("f64", "f32"), bench = true)
  es = Any[(input_resize, (Int,)), (input_set, (Int, Float64)), (output_length, ()), (output_get, (Int,))]
  for (op, args) in OPS, s in types
    op in ops && push!(es, (getfield(@__MODULE__, Symbol(op, "_", s)), args))
  end
  if bench
    for (op, args) in BENCH, s in types
      op in ops && push!(es, (getfield(@__MODULE__, Symbol(:bench_, op, "_", s)), args))
    end
  end
  es
end

end # module
