"""Minimal v2.1 verification: 1 layer SVD, orthonormality check, no training interference."""
import sys, json
from pathlib import Path
import numpy as np
from safetensors import safe_open

adapter_dir = Path("/workspace/n134/pilot/arc_s42")
config = json.loads((adapter_dir / "adapter_config.json").read_text())
scaling = config["lora_alpha"] / config["r"]
print("LoRA r=%d alpha=%d scaling=%.2f" % (config["r"], config["lora_alpha"], scaling))

# Load just ONE layer (layer 0 q_proj) - doesn't require loading all layers
path = adapter_dir / "adapter_model.safetensors"
with safe_open(str(path), framework="numpy") as f:
    keys = list(f.keys())
    # Find layer 0 q_proj
    A_key = next(k for k in keys if "lora_A" in k and "layers.0." in k and "q_proj" in k)
    B_key = next(k for k in keys if "lora_B" in k and "layers.0." in k and "q_proj" in k)
    A = f.get_tensor(A_key).astype(np.float64)
    B = f.get_tensor(B_key).astype(np.float64)

print("A key:    %s  shape=%s" % (A_key, A.shape))
print("B key:    %s  shape=%s" % (B_key, B.shape))

# Construct delta_W = scaling * B @ A and do full SVD
delta_W = scaling * B @ A  # (d_out, d_in) = (4096, 4096) for Mistral-7B q_proj
print("delta_W shape: %s  (expect (4096, 4096) for Mistral-7B q_proj)" % (delta_W.shape,))

# Fast rank-r SVD via QR
B_scaled = scaling * B
Q_B, R_B = np.linalg.qr(B_scaled, mode="reduced")
M = R_B @ A
U_small, S, Vt = np.linalg.svd(M, full_matrices=False)
U = Q_B @ U_small

# Cast to float32 (v2.1 persistence format)
U32 = U.astype(np.float32)
Vt32 = Vt.astype(np.float32)
S32 = S.astype(np.float32)

print()
print("U shape:  %s  (expect (4096, 16))" % (U32.shape,))
print("Vt shape: %s  (expect (16, 4096))" % (Vt32.shape,))
print("S shape:  %s  (expect (16,))" % (S32.shape,))

# Orthonormality checks
UtU = U32.T @ U32
u_err = float(np.abs(UtU - np.eye(16, dtype=np.float32)).max())
VVt = Vt32 @ Vt32.T
v_err = float(np.abs(VVt - np.eye(16, dtype=np.float32)).max())

# SV ordering
s_desc_err = float(max(0.0, max(np.diff(S32))))
s_pos_err = float(max(0.0, -S32.min()))

# Spot-check: reconstruct delta_W from U @ diag(S) @ Vt and compare
delta_W_recon = (U32 * S32) @ Vt32
recon_err = float(np.abs(delta_W_recon - delta_W.astype(np.float32)).max())

print()
print("ORTHONORMALITY CHECK:")
print("  max |U.T @ U - I_16|    = %.2e  (expect <1e-5 in float32)" % u_err)
print("  max |V @ V.T - I_16|    = %.2e  (expect <1e-5 in float32)" % v_err)
print()
print("SV VALIDATION:")
print("  S[0] (largest)  = %.4f" % S32[0])
print("  S[-1] (smallest) = %.4f" % S32[-1])
print("  descending err = %.2e  (expect 0.0)" % s_desc_err)
print("  non-negative err = %.2e  (expect 0.0)" % s_pos_err)
print()
print("RECONSTRUCTION CHECK:")
print("  max |U.S.Vt - delta_W|  = %.2e  (reconstruction error, expect ~1e-6)" % recon_err)

ok = u_err < 1e-3 and v_err < 1e-3 and s_desc_err < 1e-6 and s_pos_err < 1e-6
print()
print("v2.1 MINIMAL VERIFICATION: %s" % ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
