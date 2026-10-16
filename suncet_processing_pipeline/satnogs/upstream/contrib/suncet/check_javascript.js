// Run after compiling ksy/suncet.ksy with Kaitai 0.10 and installing kaitai-struct@0.10.0.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const KaitaiStream = require('kaitai-struct/KaitaiStream');
const Suncet = require('../../js/Suncet');
const frame = Buffer.from(fs.readFileSync(path.join(__dirname, 'synthetic.hex'), 'utf8').trim(), 'hex');
const expected = JSON.parse(fs.readFileSync(path.join(__dirname, 'synthetic_expected.json'), 'utf8'));
const parse = bytes => new Suncet(new KaitaiStream(bytes));
const camel = name => name.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
function repair(bytes) {
  const result = Buffer.from(bytes);
  let low = 0, high = 0;
  for (let i = 0; i < 124; i++) {
    const word = result.readUInt16LE(16 + 2 * i);
    low += word;
    high += (124 - i) * word;
  }
  result.writeUInt32BE((high % 65535) * 65536 + (low % 65535), 264);
  return result;
}
const beacon = parse(frame).beacon;
for (const [name, value] of Object.entries(expected)) {
  const actual = beacon[camel(name)];
  assert.equal(typeof actual, 'number', name);
  assert.ok(Math.abs(actual - value) <= Math.max(1e-12, Math.abs(value) * 1e-12), name);
}
for (let i = 0; i < frame.length; i++) {
  const corrupt = Buffer.from(frame);
  corrupt[i] ^= 1;
  assert.throws(() => parse(corrupt), undefined, `protected byte ${i}`);
}
for (const wrong of [frame.subarray(16), frame.subarray(0, 267), Buffer.concat([frame, Buffer.from([0, 0])])]) {
  assert.throws(() => parse(wrong));
}
for (const [offset, value] of [[0, 0x0802], [0, 0x1801], [4, 244], [10, 1000]]) {
  const bad = Buffer.from(frame);
  bad.writeUInt16BE(value, 16 + offset);
  assert.throws(() => parse(repair(bad)));
}
for (const value of [NaN, Infinity, -Infinity]) {
  const bad = Buffer.from(frame);
  bad.writeFloatBE(value, 16 + 68);
  assert.ok(Number.isNaN(parse(repair(bad)).beacon.dspsFlareLevel));
}
// Explicitly exercise a checksum above 2^31 to detect signed-JS arithmetic bugs.
let highChecksum;
for (let byte = 0; byte <= 255; byte++) {
  const candidate = Buffer.from(frame);
  candidate[16 + 247] = byte;
  const repaired = repair(candidate);
  if (repaired.readUInt32BE(264) >= 2 ** 31) { highChecksum = repaired; break; }
}
assert.ok(highChecksum);
parse(highChecksum);
console.log('SunCET JavaScript checks passed: 111 fields, all 268 protected bytes, envelope, nonfinite data, unsigned checksum.');
