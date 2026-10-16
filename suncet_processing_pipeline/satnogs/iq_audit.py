"""Offline audit of a SunCET complex-float32 IQ recording.

Uses only file/vector sources and receive DSP: no UHD, transmit, or socket
blocks. The JSON report contains aggregate evidence, never packet bytes,
decoded telemetry values, or other-APID identifiers. Raw IQ and even raw APID 1
frames still contain information excluded from the approved public interface.

Run in the suncet conda environment (GNU Radio, NumPy and SciPy required):
python -m suncet_processing_pipeline.satnogs.iq_audit INPUT --output REPORT
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .ax25 import crc16_x25, decode_public_ax25_beacon, SUNCET_AX25_HEADER
from .beacon_contract import BeaconValidationError

FLAG = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)


def descramble(bits):
    """G3RUH multiplicative descrambler: b[n] XOR b[n-12] XOR b[n-17]."""
    result = np.asarray(bits, dtype=np.uint8).copy()
    result[12:] ^= bits[:-12]
    result[17:] ^= bits[:-17]
    return result


def nrzi_decode(bits):
    """No transition represents 1. Initial previous level is zero."""
    result = np.empty_like(bits)
    if len(bits):
        result[0] = 1 ^ bits[0]
        result[1:] = 1 ^ bits[1:] ^ bits[:-1]
    return result


def flag_positions(bits):
    if len(bits) < 8:
        return np.array([], dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(bits, 8)
    return np.flatnonzero(np.all(windows == FLAG, axis=1))


def recover_hdlc_frames(bits):
    """Yield CRC-valid frames, retaining FCS, between exact HDLC flags.

    Each tuple contains opening/closing bit offsets, bytes, and stuffed-zero
    count. Offsets refer to recovered symbols, not exact IQ sample positions.
    Reject aborts, bad byte alignment, invalid CRCs, and oversized candidates.
    """
    positions = flag_positions(bits)
    for left, right in zip(positions[:-1], positions[1:]):
        stuffed = bits[left + 8:right]
        if not 48 <= len(stuffed) <= 4820:
            continue
        data = []
        ones = removed = 0
        valid = True
        for bit in stuffed:
            if ones == 5:
                if bit:
                    valid = False
                    break
                removed += 1
                ones = 0
                continue
            data.append(bit)
            ones = ones + 1 if bit else 0
        if not valid or len(data) % 8:
            continue
        frame = np.packbits(data, bitorder="little").tobytes()
        if len(frame) < 6 or len(frame) > 502:
            continue
        if crc16_x25(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
            continue
        yield int(left), int(right + 8), frame, removed


def is_public_beacon(frame, *, fcs_present):
    try:
        values = decode_public_ax25_beacon(frame, fcs_present=fcs_present)
    except BeaconValidationError:
        return False
    assert len(values) == 111
    return True


def demodulate(baseband):
    """Rebuild the supplied flowgraph's GFSK branch using GNU Radio 3.10.

    GNU Radio 3.10's gfsk_demod uses symbol_sync_ff (M&M), whereas the supplied
    GNU Radio 3.8 flowgraph used the older clock-recovery implementation.
    """
    from gnuradio import gr, blocks, digital
    import pmt

    flow = gr.top_block()
    source = blocks.vector_source_c(baseband, False)
    demod = digital.gfsk_demod(
        samples_per_symbol=10, sensitivity=math.pi / 10, gain_mu=0.175,
        mu=0.5, omega_relative_limit=0.005, freq_error=0.0,
        verbose=False, log=False,
    )
    scrambler = digital.descrambler_bb(0x21, 0, 16)
    differential = digital.diff_decoder_bb(2)
    invert = digital.map_bb([1, 0])
    sliced = blocks.vector_sink_b()
    hdlc_bits = blocks.vector_sink_b()
    deframer = digital.hdlc_deframer_bp(4, 500)
    messages = blocks.message_debug()
    flow.connect(source, demod, scrambler, differential, invert, hdlc_bits)
    flow.connect(demod, sliced)
    flow.connect(invert, deframer)
    flow.msg_connect(deframer, "out", messages, "store")
    flow.run()
    frames = [bytes(pmt.u8vector_elements(pmt.cdr(messages.get_message(i))))
              for i in range(messages.num_messages())]
    return (np.asarray(sliced.data(), dtype=np.uint8),
            np.asarray(hdlc_bits.data(), dtype=np.uint8), frames, gr.version())


def audit(path, sample_rate=307200):
    from scipy.signal import firwin, lfilter, resample_poly
    from scipy.ndimage import minimum_filter1d, maximum_filter1d

    if sample_rate != 307200:
        raise ValueError("This validated recipe expects 307200 complex samples/s")
    raw = path.read_bytes()
    if len(raw) % 8 or not raw:
        raise ValueError("Input must contain complete little-endian complex float32 samples")
    iq = np.frombuffer(raw, dtype="<c8")
    if not np.isfinite(iq).all():
        raise ValueError("Non-finite IQ input")
    # Defined offline approximation of the supplied 9 kHz receive filter and
    # 307.2 -> 96 ksample/s resampler; not a bit-identical replay of the GRC.
    filtered = lfilter(firwin(257, 9000, fs=sample_rate), [1], iq)
    baseband = resample_poly(filtered, 5, 16).astype(np.complex64)
    sliced, bits, gr_frames, version = demodulate(baseband)
    assert np.array_equal(bits, nrzi_decode(descramble(sliced)))
    # The two linear operations commute after startup transients.
    alternate = descramble(nrzi_decode(sliced))
    assert np.array_equal(bits[18:], alternate[18:])
    frames = list(recover_hdlc_frames(bits))
    assert Counter(f[:-2] for _, _, f, _ in frames) == Counter(gr_frames)
    public = [f for f in frames if is_public_beacon(f[2], fcs_present=True)]
    assert sum(is_public_beacon(f, fcs_present=False) for f in gr_frames) == len(public)
    flags = set(flag_positions(bits))
    deviations, offsets, measurements = [], [], []
    for left, right, frame, removed in public:
        pre = post = 1
        at = left - 8
        while at in flags:
            pre += 1
            at -= 8
        at = right
        while at in flags:
            post += 1
            at += 8
        # Approximate bit-clock positions suffice for interior waveform
        # statistics. They must never be used as a privacy-safe IQ crop boundary.
        a, b = int((left / 9600 + .015) * 96000), int((right / 9600 - .015) * 96000)
        signal = baseband[a:b]
        freq = np.angle(signal[1:] * signal[:-1].conj()) * 96000 / (2 * np.pi)
        negative = maximum_filter1d(freq, size=31) < 0
        positive = minimum_filter1d(freq, size=31) > 0
        if negative.any() and positive.any():
            low, high = float(np.median(freq[negative])), float(np.median(freq[positive]))
            deviations.append((high - low) / 2)
            offsets.append((high + low) / 2)
        measurements.append({
            "approx_start_seconds": round(left / 9600, 6),
            "approx_end_seconds": round(right / 9600, 6),
            "frame_bytes_including_fcs": len(frame), "stuffed_zero_bits": removed,
            "consecutive_opening_flags": pre, "consecutive_closing_flags": post,
        })
    # Tests of simpler hypotheses use the same recovered symbol decisions.
    controls = {
        "without_descrambling": len(list(recover_hdlc_frames(nrzi_decode(sliced)))),
        "without_nrzi_decoding": len(list(recover_hdlc_frames(descramble(sliced)))),
        "nrzi_before_descrambling": len(list(recover_hdlc_frames(alternate))),
    }
    return {
        "input": {"file": path.name, "size_bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(),
                  "format": "little-endian complex float32 (interleaved I/Q)",
                  "complex_samples": len(iq), "sample_rate_hz": sample_rate,
                  "duration_seconds": len(iq) / sample_rate},
        "receiver": {"gnu_radio_version": version, "baud": 9600,
                     "baseband_sample_rate_hz": 96000, "lowpass_cutoff_hz": 9000,
                     "fir_taps": 257, "fir_window": "hamming",
                     "resampling_up_down": [5, 16], "frequency_translation_hz": 0,
                     "gfsk_sensitivity": math.pi / 10, "clock_gain_mu": .175,
                     "clock_relative_limit": .005,
                     "descrambler_mask": "0x21", "descrambler_seed": 0,
                     "descrambler_length_parameter": 16,
                     "nrzi": "1 for no transition", "hdlc_byte_bit_order": "LSB first",
                     "fec_or_deinterleaving_applied": False},
        "results": {"independent_crc_valid_frames": len(frames),
                    "gnu_radio_crc_valid_frames": len(gr_frames),
                    "frame_multisets_match_after_fcs_removal": True,
                    "public_beacons_both_checksums_valid": len(public),
                    "other_crc_valid_frames_excluded": len(frames) - len(public),
                    "public_fields_per_beacon": 111,
                    "header_matches": all(f[2].startswith(SUNCET_AX25_HEADER) for f in public),
                    "beacon_frames_unique": len({f[2] for f in public}),
                    "control_crc_valid_counts": controls},
        "waveform_estimate": {
            "method": "Median positive/negative frequency plateaus within >=31-sample same-sign windows at 96 ksample/s; interior of validated beacon frames",
            "deviation_hz_median": round(float(np.median(deviations)), 1) if deviations else None,
            "deviation_hz_per_beacon": [round(v, 1) for v in deviations],
            "baseband_offset_hz_median": round(float(np.median(offsets)), 1) if offsets else None,
            "caveat": "Receiver-filtered estimate, not calibrated absolute RF frequency or radio configuration readback; Gaussian BT not established",
        },
        "beacon_framing": measurements,
        "privacy": "No IQ samples, packet bytes, telemetry values, or other packet identifiers exported. Approximate boundaries are not approved sharing crops.",
        "scope": "Offline receive reconstruction; actual SatNOGS Network flowgraph, 19200-baud mode, and paired independent reception not tested",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--sample-rate", type=int, default=307200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input, args.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
