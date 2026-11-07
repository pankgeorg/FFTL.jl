// node check.mjs [fftl.wasm] [reference.json]: run each case of reference.json (wasm/reference.jl: native FFTL) with
// the module and compare. Relative difference = max |wasm - native| / max |native|. EXACT: bitwise equal; CLOSE:
// within 1e-12 (Float64) or 1e-5 (Float32); otherwise BAD. Needs julia.mjs (WasmCodegen's JavaScript host, in the
// release) next to this file.
import fs from 'node:fs';
import { instantiate } from './julia.mjs';

const dir = new URL('.', import.meta.url).pathname;
const [wasm = dir + 'fftl.wasm', ref = dir + 'reference.json'] = process.argv.slice(2);
const t0 = performance.now();
const { instance } = await instantiate(fs.readFileSync(wasm));
const e = instance.exports;
console.log(`instantiated ${wasm} in ${(performance.now() - t0).toFixed(0)} ms`);

const counts = { EXACT: 0, CLOSE: 0, BAD: 0 };
const cases = JSON.parse(fs.readFileSync(ref, 'utf8'));
for (const { op, dims, status, fftw, input, output } of cases) {
  e.input_resize(BigInt(input.length));
  input.forEach((x, i) => e.input_set(BigInt(i + 1), x));
  const t = performance.now();
  let st, got;
  try {
    st = Number(e[op](...dims.map(BigInt)));
    got = Array.from({ length: Number(e.output_length()) }, (_, i) => e.output_get(BigInt(i + 1)));
  } catch (err) {
    counts.BAD++;
    console.log(`BAD   ${op}(${dims.join(', ')}): ${err}`);
    continue;
  }
  const ms = performance.now() - t;
  let diff = 0, scale = 0, same = got.length === output.length;
  for (let i = 0; i < output.length; i++) {
    diff = Math.max(diff, Math.abs(got[i] - output[i])); scale = Math.max(scale, Math.abs(output[i]));
    if (!Object.is(got[i], output[i])) same = false;
  }
  const rel = got.length === output.length ? diff / (scale || 1) : Infinity;
  const tol = op.endsWith('f32') ? 1e-5 : 1e-12;
  const v = st !== status ? 'BAD' : same ? 'EXACT' : rel <= tol ? 'CLOSE' : 'BAD';
  counts[v]++;
  console.log(`${v.padEnd(5)} ${op}(${dims.join(', ')}): relative difference ${rel.toExponential(2)} (native FFTW: ${fftw.toExponential(2)}), ${ms.toFixed(1)} ms`);
}
console.log(`${cases.length} cases: ${counts.EXACT} EXACT, ${counts.CLOSE} CLOSE, ${counts.BAD} BAD`);
process.exit(counts.BAD ? 1 : 0);
