"""
Compatibility alias.

Ensures `import protocol` returns the *same module object* as `gradience.bench.protocol`,
so mocks/patching are consistent across environments.
"""
import sys as _sys
from gradience.bench import protocol as _protocol

_sys.modules[__name__] = _protocol