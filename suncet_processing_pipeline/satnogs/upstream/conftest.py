"""Contribution tests run in satnogs-decoders after its Kaitai compilation.

Keep the mission repository's pytest discovery from importing tests that need
the separate upstream package/runtime. This file is not part of the six-file
upstream contribution; its test suite discovers tests/test_suncet.py normally.
"""

collect_ignore = ["tests"]
