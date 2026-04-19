"""Minimal v2.1 verification: 1 layer SVD, orthonormality check, no training interference."""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

adapter_dir = Path("/workspace/n134/pilot/arc_s42")
config = json.loads((adapter_dir / "adapter_config.json").read_text())
scaling = config["lora_alpha"] / config["r"]
print(f"LoRA r={config['r']} alpha={config['lora_alpha']} scaling={scaling:.2f}")

# Load just ONE layer (layer 0 q_proj) - doesn't require loading all layers
path = adapter_dir / "adapter_model.safetensors"
with safe_open(str(path), framework="numpy") as f:
    keys = list(f.keys())
    # Find layer 0 q_proj
    A_key = next(k for k in keys if "lora_A" in k and "layers.0." in k and "q_proj" in k)
    B_key = next(k for k in keys if "lora_B" in k and "layers.0." in k and "q_proj" in k)
    A = f.get_tensor(A_key).astype(np.float64)
    B = f.get_tensor(B_key).astype(np.float64)

print(f"A key:    {A_key}  shape={A.shape}")
print(f"B key:    {B_key}  shape={B.shape}")

# Construct delta_W = scaling * B @ A and do full SVD
delta_W = scaling * B @ A  # (d_out, d_in) = (4096, 4096) for Mistral-7B q_proj
print(f"delta_W shape: {delta_W.shape}  (expect (4096, 4096) for Mistral-7B q_proj)")

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
print(f"U shape:  {U32.shape}  (expect (4096, 16))")
print(f"Vt shape: {Vt32.shape}  (expect (16, 4096))")
print(f"S shape:  {S32.shape}  (expect (16,))")

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
print(f"  max |U.T @ U - I_16|    = {u_err:.2e}  (expect <1e-5 in float32)")
print(f"  max |V @ V.T - I_16|    = {v_err:.2e}  (expect <1e-5 in float32)")
print()
print("SV VALIDATION:")
print(f"  S[0] (largest)  = {S32[0]:.4f}")
print(f"  S[-1] (smallest) = {S32[-1]:.4f}")
print(f"  descending err = {s_desc_err:.2e}  (expect 0.0)")
print(f"  non-negative err = {s_pos_err:.2e}  (expect 0.0)")
print()
print("RECONSTRUCTION CHECK:")
print(f"  max |U.S.Vt - delta_W|  = {recon_err:.2e}  (reconstruction error, expect ~1e-6)")

ok = u_err < 1e-3 and v_err < 1e-3 and s_desc_err < 1e-6 and s_pos_err < 1e-6
print()
print("v2.1 MINIMAL VERIFICATION: %s" % ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
