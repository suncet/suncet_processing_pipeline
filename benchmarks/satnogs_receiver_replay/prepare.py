"""Build an isolated, receive-only subset of pinned upstream SatNOGS sources.

Run with the suncet Python on macOS. See README.md for source downloads.
No source checkout or conda installation is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import yaml

GR_COMMIT = "4defcfc445c1f94450c48ad348689e7986ebe6f7"  # v3.1.0.1
FLOWGRAPHS_COMMIT = "75eb7300fdf38ca3e6455d8bed687fc10d680054"  # 2.7
CRCPP_COMMIT = "6da4f42ec110c22d10973f4c16927700f1b52bf1"


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args])


def receive_subgraph(original):
    """Keep the exact upstream demodulator blocks/edges; replace only I/O seams."""
    graph = yaml.safe_load(original)
    by_name = {b["name"]: b for b in graph["blocks"]}
    # Walk upstream from the bit slicer, stopping at the baseband input seam.
    names = {"digital_binary_slicer_fb_0"}
    while True:
        parents = {edge[0] for edge in graph["connections"] if edge[2] in names}
        new = names | parents
        if new == names:
            break
        names = new
    assert "virtual_source_0" in names
    names |= {"audio_samp_rate", "baudrate", "decimation", "import_0", "import_1"}
    kept = [b for b in graph["blocks"] if b["name"] in names]
    for block in kept:
        if block["id"] == "import":
            # The generic flowgraph also imports unused Hamlib for the removed rig.
            text = block["parameters"]["imports"]
            block["parameters"]["imports"] = text.replace("import Hamlib", "")
    source = by_name["virtual_source_0"]
    source["id"] = "pad_source"
    source["parameters"] = {"label": "in", "type": "complex", "vlen": "1",
                            "num_streams": "1", "optional": "False"}
    sink = {"name": "bits_out", "id": "pad_sink",
            "parameters": {"label": "bits", "type": "byte", "vlen": "1",
                           "num_streams": "1", "optional": "False"},
            "states": {"state": "enabled", "coordinate": [1800, 300], "rotation": 0}}
    satnogs_import = {"name": "upstream_utils_import", "id": "import",
                      "parameters": {"imports": "import gnuradio.satnogs as satnogs"},
                      "states": {"state": "enabled", "coordinate": [0, 0], "rotation": 0}}
    graph["blocks"] = kept + [sink, satnogs_import]
    graph["connections"] = [e for e in graph["connections"]
                            if e[0] in names and e[2] in names]
    graph["connections"].append(["digital_binary_slicer_fb_0", "0", "bits_out", "0"])
    options = graph["options"]["parameters"]
    options.update(id="suncet_satnogs_fsk_rx", generate_options="hb",
                   title="Offline SatNOGS FSK receive subgraph")
    return graph


def prepare(root):
    root = root.resolve()
    src = root / "gr-satnogs"
    flowgraphs = root / "satnogs-flowgraphs"
    if git(src, "rev-parse", "HEAD").decode().strip() != GR_COMMIT:
        raise ValueError("gr-satnogs must be checked out at the pinned commit")
    if git(src, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("gr-satnogs tracked sources must be unmodified")
    if git(src / "CRCpp", "rev-parse", "HEAD").decode().strip() != CRCPP_COMMIT:
        raise ValueError("Initialize the pinned CRCpp submodule")
    original = git(flowgraphs, "show", f"{FLOWGRAPHS_COMMIT}:generic/fsk_ax25.grc")
    dispatcher = git(flowgraphs, "show", f"{FLOWGRAPHS_COMMIT}:flowgraph_dispatcher/flowgraph_dispatcher.py")
    # Read the mode map without importing/executing the network dispatcher.
    import ast
    tree = ast.parse(dispatcher)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {
                "SATNOGS_FLOWGRAPH_SCRIPTS", "SATNOGS_FLOWGRAPH_MODES"}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "mode_map", "exec"),
                 {"__builtins__": {}}, constants)
    mode = constants["SATNOGS_FLOWGRAPH_MODES"]["GFSK"]
    assert mode["script_name"] == "satnogs_fsk_ax25.py" and mode["framing"] == "ax25"
    graph = receive_subgraph(original)
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    grc = generated / "receive.grc"
    grc.write_text(yaml.safe_dump(graph, sort_keys=False))

    # Make only the upstream utility available to GRC's variable evaluator.
    spec = importlib.util.spec_from_file_location(
        "gnuradio.satnogs", src / "python/satnogs/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    sys.modules["gnuradio.satnogs"] = utils
    from gnuradio.grc.core import Constants
    from gnuradio.grc.compiler import main as grcc
    Constants.CACHE_FILE = str(root / "grc-cache.json")
    grcc(argparse.Namespace(output=str(generated), user_lib_dir=False, run=False,
                           grc_files=[str(grc)]))

    # Apple Clang 21 rejects Boost 1.78's out-of-range enum arithmetic in MPL.
    # A local header overlay gives the two enums explicit int underlying types.
    # Their values are unchanged; no GNU Radio or SatNOGS source is patched.
    prefix = Path(sys.prefix)
    compat = root / "compiler-compat"
    for name in ("udt_builtin_mixture_enum", "int_float_mixture_enum"):
        relative = Path("boost/numeric/conversion") / (name + ".hpp")
        text = (prefix / "include" / relative).read_text()
        text = text.replace(f"enum {name}\n", f"enum {name} : int\n")
        target = compat / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    bridge = Path(__file__).with_name("bridge.cc").resolve()
    cpp_names = ["ax25_decoder.cc", "decoder.cc", "frame_decoder_impl.cc",
                 "metadata.cc", "doppler_correction_cc_impl.cc", "dummy_doppler_correction.cc"]
    cmd = ["clang++", "-std=c++17", "-O2", "-shared", "-undefined", "dynamic_lookup", "-fPIC",
           '-DPYBIND11_COMPILER_TYPE=""', '-DPYBIND11_BUILD_ABI=""',
           "-Dgnuradio_satnogs_EXPORTS", "-DSPDLOG_FMT_EXTERNAL", "-DSPDLOG_COMPILED_LIB",
           "-Wno-deprecated-declarations", "-Wno-deprecated-literal-operator", "-Wno-deprecated-builtins"]
    # Match the installed conda GNU Radio pybind11 ABI (v4_libcpp).
    for include in [compat, prefix / "include", Path(sysconfig.get_path("include")),
                    root / "include", root / "build-deps/pybind11/include",
                    src / "include", src / "CRCpp/inc"]:
        cmd.extend(["-I", str(include)])
    cmd += [str(bridge)] + [str(src / "lib" / name) for name in cpp_names]
    cmd += ["-L", str(prefix / "lib"), "-Wl,-rpath," + str(prefix / "lib"),
            "-lgnuradio-runtime", "-lgnuradio-pmt", "-lgnuradio-digital", "-lvolk", "-lspdlog", "-lfmt",
            "-o", str(root / ("suncet_receiver_native" + sysconfig.get_config_var("EXT_SUFFIX")))]
    with (root / "build.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = {"gr_satnogs_commit": GR_COMMIT, "satnogs_flowgraphs_commit": FLOWGRAPHS_COMMIT,
                "crcpp_commit": CRCPP_COMMIT, "mode": mode,
                "native_sources": cpp_names,
                "original_grc_sha256": hashlib.sha256(original).hexdigest(),
                "generated_grc_sha256": hashlib.sha256(grc.read_bytes()).hexdigest(),
                "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest()}
    (root / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Offline receive blocks built; no hardware, network, or transmit blocks included.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("tmp/satnogs-receiver"))
    prepare(parser.parse_args().workdir)
