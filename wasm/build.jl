# julia --project=wasm wasm/build.jl [OUT.wasm]: the prebuilt module of wasm/fft.jl's transforms (see WASM.md),
# optimized by wasm-opt -O3 (WasmCodegen's default). ENV: OPS=fft,rfft,... and TYPES=f64,f32 build a subset;
# BENCH=0 leaves out the bench_* exports.
using WasmCodegen, FFTL, WasmCompat, SIMD
Base.get_extension(FFTL, :FFTLWasmCodegenExt) === nothing && error("FFTLWasmCodegenExt is not loaded")
include(joinpath(@__DIR__, "fft.jl"))
out = get(ARGS, 1, joinpath(@__DIR__, "fftl.wasm"))
ops = haskey(ENV, "OPS") ? Symbol.(split(ENV["OPS"], ',')) : first.(FFTLWasm.OPS)
types = haskey(ENV, "TYPES") ? String.(split(ENV["TYPES"], ',')) : ["f64", "f32"]
config = WasmCodegen.Config(; trim = :safe, on_unsupported = :error, progress = true)
config.optimize === nothing && error("modules are built optimized (wasm-opt -O3) only")
t = time()
c = WasmCodegen.compile(FFTLWasm.entries(; ops, types, bench = get(ENV, "BENCH", "1") != "0"); config)
write(out, c.bytes)
println("wrote ", out, ": ", length(c.bytes), " bytes in ", round(Int, time() - t), " s")
