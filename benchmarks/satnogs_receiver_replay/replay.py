"""Replay private flight IQ through pinned upstream SatNOGS receive blocks.

Only aggregate validation evidence is saved. All decoded frames, including
non-beacons and the opaque parts of APID 1, remain in process memory.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

# Allow running directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_receiver(root):
    sys.path[:0] = [str(root), str(root / "generated")]
    import suncet_receiver_native as native
    sys.modules["gnuradio.satnogs"] = native
    source = root / "gr-satnogs/python/satnogs"
    spec = importlib.util.spec_from_file_location("upstream_utils", source / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    native.find_decimation = utils.find_decimation
    # Hamlib is an unused import in this upstream Python file. Remove only that
    # import; retain every filter/resampler and connect the upstream zero-offset
    # dummy Doppler source in place of the live rigctl source.
    path = source / "doppler_compensation.py"
    doppler = types.ModuleType("offline_doppler_compensation")
    exec(compile(path.read_text().replace("import Hamlib\n", ""), str(path), "exec"),
         doppler.__dict__)
    from suncet_satnogs_fsk_rx import suncet_satnogs_fsk_rx
    return native, doppler.doppler_compensation, suncet_satnogs_fsk_rx


def receive(iq, receiver, descramble=True):
    from gnuradio import gr, blocks
    import pmt
    native, compensate, fsk_type = receiver
    flow = gr.top_block("SunCET offline SatNOGS release replay")
    source = blocks.vector_source_c(iq, False)
    doppler = compensate(native.dummy_doppler_correction.make(0),
                         307200, 401200000, 0, 57600, False)
    demod = fsk_type(baudrate=9600)
    decoder = native.frame_decoder(
        native.ax25_decoder.make("GND", 0, descramble, True, 1024, False), 1)
    messages = blocks.message_debug()
    flow.connect(source, doppler, demod, decoder)
    flow.msg_connect(decoder, "out", messages, "store")
    flow.run()
    frames = []
    for i in range(messages.num_messages()):
        # Exercise the PMT transport serialization without opening ZMQ sockets.
        message = pmt.deserialize_str(pmt.serialize_str(messages.get_message(i)))
        metadata = pmt.to_python(pmt.car(message))
        assert metadata["decoder_crc_valid"] is True
        frame = pmt.to_python(pmt.cdr(message)).tobytes()
        # Mirror the release subscriber's JSON/base64 representation in memory.
        metadata["pdu"] = base64.b64encode(frame).decode("utf-8")
        saved = json.loads(json.dumps(metadata))
        restored = base64.b64decode(saved["pdu"], validate=True)
        assert restored == frame
        frames.append(restored)
    return frames


def approved(frames):
    from suncet_processing_pipeline.satnogs.ax25 import decode_public_ax25_beacon
    from suncet_processing_pipeline.satnogs.beacon_contract import BeaconValidationError

    result = []
    for frame in frames:
        try:
            fields = decode_public_ax25_beacon(frame, fcs_present=False)
        except BeaconValidationError:
            continue
        assert len(fields) == 111
        result.append(frame)
    return result


def replay(path, root):
    from scipy.signal import firwin, lfilter, resample_poly
    from gnuradio import gr
    from suncet_processing_pipeline.satnogs.iq_audit import demodulate, recover_hdlc_frames
    raw = path.read_bytes()
    if not raw or len(raw) % 8:
        raise ValueError("Expected complete little-endian complex float32 samples")
    iq = np.frombuffer(raw, dtype="<c8")
    if not np.isfinite(iq).all():
        raise ValueError("Non-finite IQ input")
    receiver = load_receiver(root)
    frames = receive(iq, receiver)
    control = receive(iq, receiver, descramble=False)
    # Independently repeat the previously validated receive reconstruction and
    # NumPy HDLC/FCS deframer on this same IQ; compare actual bytes, not only counts.
    filtered = lfilter(firwin(257, 9000, fs=307200), [1], iq)
    baseband = resample_poly(filtered, 5, 16).astype(np.complex64)
    _, bits, reference, _ = demodulate(baseband)
    independent = [f[:-2] for _, _, f, _ in recover_hdlc_frames(bits)]
    assert Counter(reference) == Counter(independent)
    public = approved(frames)
    reference_public = approved(reference)
    results = {
        "crc_valid_frames": len(frames),
        "public_beacons": len(public),
        "distinct_public_beacons": len(set(public)),
        "other_frames_excluded": len(frames) - len(public),
        "public_fields_per_beacon": 111 if public else None,
        "public_frame_lengths_bytes": sorted({len(f) for f in public}),
        "all_frames_match_reference_bytes": Counter(frames) == Counter(reference),
        "all_beacons_match_reference_bytes": Counter(public) == Counter(reference_public),
        "reference_matches_independent_hdlc_bytes": True,
        "pmt_and_json_transport_preserve_bytes": True,
        "crc_valid_frames_without_descrambling": len(control),
    }
    # Fail closed: zero/partial reception is not a passing validation report.
    if not public or not results["all_frames_match_reference_bytes"] or control:
        raise RuntimeError("Receiver replay did not satisfy the validation contract")
    manifest = json.loads((root / "build-manifest.json").read_text())
    return {"input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_bytes": len(raw), "sample_rate_hz": 307200,
            "gnu_radio_version": gr.version(), "upstream_build": manifest,
            "receiver": {"mode": "GFSK", "baud": 9600,
                         "flowgraph": "satnogs_fsk_ax25.py",
                         "baseband_rate_hz": 57600, "quadrature_gain": 1.2,
                         "g3ruh_descrambling": True, "crc_check": True,
                         "one_bit_correction": False, "doppler_offset_hz": 0,
                         "sun_cet_adapter_fcs_present": False},
            "results": results,
            "scope": "Local replay of released receiver blocks. No live station, network upload, upstream telemetry ingestion, or 19200-baud validation.",
            "privacy": "No IQ samples, packet bytes, telemetry values, or other packet identifiers exported."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("tmp/satnogs-receiver"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.input, args.workdir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
