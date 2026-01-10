import os
import unittest

ARTIFACTS = ("PYint", "PYr")

class TestNoPatchArtifacts(unittest.TestCase):
    def test_no_patch_artifacts_in_gradience(self):
        root = os.path.join(os.path.dirname(__file__), "..", "gradience")
        root = os.path.abspath(root)

        offenders = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                # Skip obvious binary-ish files
                if fn.endswith((".pyc", ".pkl", ".bin", ".pt", ".pth", ".safetensors")):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue

                for a in ARTIFACTS:
                    if a in text:
                        offenders.append((path, a))
        if offenders:
            msg = "\n".join([f"{p} contains {a}" for p, a in offenders[:50]])
            self.fail("Patch artifacts found in gradience/ tree:\n" + msg)

if __name__ == "__main__":
    unittest.main()
