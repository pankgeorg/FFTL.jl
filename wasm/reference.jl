# julia --project=wasm wasm/reference.jl [OUT.json]: wasm/fft.jl's transforms run natively (FFTL's normal plans) on
# fixed inputs: the reference wasm/check.mjs compares a module with. Each case also records FFTW's result's relative
# difference from FFTL's (`fftw`), for scale.
import FFTL, FFTW
using LinearAlgebra
include(joinpath(@__DIR__, "fft.jl"))
const W = FFTLWasm
out = get(ARGS, 1, joinpath(@__DIR__, "reference.json"))

# fixed inputs (Float64 values exactly representable in Float32, so both precisions read the same numbers)
cdata(len, s) = Float64.(Float32.([isodd(i) ? sin(0.37i + s) : cos(0.23i + 2s) for i in 1:2len]))
rdata(len, s) = Float64.(Float32.([sin(0.37i + s) + 0.25cos(1.7i) for i in 1:len]))
cplx(v, T) = Complex{T}.(v[1:2:end], v[2:2:end])
flat(A::AbstractArray{<:Complex}) = Float64.(vec(reinterpret(real(eltype(A)), vec(A))))
flat(A::AbstractArray{<:Real}) = Float64.(vec(A))
reldiff(a, b) = maximum(abs, a .- b; init = 0.0) / max(maximum(abs, b; init = 0.0), floatmin())

function fftw_result(op, T, dims, input)
  op === :fft  && return FFTW.fft(cplx(input, T))
  op === :bfft && return FFTW.bfft(cplx(input, T))
  op === :ifft && return FFTW.ifft(cplx(input, T))
  op === :fft2 && return FFTW.fft(reshape(cplx(input, T), dims...))
  op === :ifft2 && return FFTW.ifft(reshape(cplx(input, T), dims...))
  op === :rfft && return FFTW.rfft(T.(input))
  op === :irfft && return FFTW.irfft(cplx(input, T), dims[1])
  op === :dct  && return FFTW.dct(T.(input))
  op === :idct && return FFTW.idct(T.(input))
end

function case(io, op, s, T, dims, input)
  W.input_resize(length(input))
  for (i, x) in enumerate(input); W.input_set(i, x); end
  status = getfield(W, Symbol(op, "_", s))(dims...)
  output = copy(W.OUT)
  fw = reldiff(flat(fftw_result(op, T, dims, input)), output)
  print(io, "{\"op\":\"", op, "_", s, "\",\"dims\":[", join(dims, ","), "],\"status\":", status, ",\"fftw\":", fw,
        ",\"input\":[", join(repr.(input), ","), "],\"output\":[", join(repr.(output), ","), "]}")
  println(rpad("$(op)_$s$(dims)", 24), " FFTW vs FFTL: ", fw)
end

open(out, "w") do io
  print(io, "[\n")
  first = true
  sep() = (first || print(io, ",\n"); first = false)
  for (s, T) in (("f64", Float64), ("f32", Float32))
    # powers of 2, a smooth composite, and primes (Bluestein: 1009 > the largest codelet)
    for n in (8, 64, 1024, 1000, 13, 1009)
      sep(); case(io, :fft, s, T, (n,), cdata(n, 0.1))
    end
    sep(); case(io, :bfft, s, T, (60,), cdata(60, 0.2))
    for n in (64, 97)
      sep(); case(io, :ifft, s, T, (n,), cdata(n, 0.3))
    end
    for (m, n) in ((16, 16), (12, 10), (64, 32), (7, 11))
      sep(); case(io, :fft2, s, T, (m, n), cdata(m * n, 0.4))
    end
    sep(); case(io, :ifft2, s, T, (32, 24), cdata(32 * 24, 0.5))
    for n in (64, 1000, 101)
      sep(); case(io, :rfft, s, T, (n,), rdata(n, 0.6))
    end
    for n in (64, 30, 101)
      sep(); case(io, :irfft, s, T, (n,), cdata(n ÷ 2 + 1, 0.7))
    end
    for n in (8, 64, 100, 37)
      sep(); case(io, :dct, s, T, (n,), rdata(n, 0.8))
    end
    sep(); case(io, :idct, s, T, (64,), rdata(64, 0.9))
  end
  print(io, "\n]\n")
end
println("wrote ", out)
