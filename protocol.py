"""
Compatibility shim.

Some legacy bench/UDR code paths and tests expect `import protocol` to work.
Map that to the canonical implementation in `gradience.bench.protocol`.
"""
from gradience.bench.protocol import *  # noqa: F401,F403