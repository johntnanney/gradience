# Guard Critical Warnings

## What Guard IS

✅ **Emergency Brake**: Stops runaway training when numerical instability detected  
✅ **Time Machine**: Rolls back LoRA adapter weights to earlier states  
✅ **Safety Net**: Catches gradient explosions and NaN/Inf corruption  
✅ **EXPERIMENTAL**: Not recommended for production use  

## What Guard is NOT

❌ **NOT a Fix**: Cannot repair data bugs or bad objectives  
❌ **NOT a Solver**: Won't fix hyperparameter problems  
❌ **NOT Magic**: Rolling back doesn't mean better model quality  
❌ **NOT Complete**: Only protects LoRA adapters, not full models  

## Critical Warnings

### 🚨 Guard CAN and WILL:
1. **STOP your training** - Halts when protection limits exceeded
2. **ROLL BACK weights** - Reverts adapter parameters without asking
3. **MISS root causes** - Treats symptoms, not diseases
4. **FAIL to help** - May not improve final model quality

### 🚨 You MUST:
1. **VALIDATE with eval** - Always check if Guard actually helped
2. **FIX root causes** - Investigate data, objectives, hyperparameters
3. **MONITOR telemetry** - Watch for repeated triggers
4. **DISABLE if unhelpful** - Turn off if it doesn't improve outcomes

## Telemetry Severity Levels

Guard uses consistent severity levels to communicate state:

| Severity | Meaning | Example Events |
|----------|---------|----------------|
| **INFO** | Normal operation | Snapshot taken, trigger detected |
| **WARNING** | ACTION TAKEN | **Weights rolled back** |
| **ERROR** | PROTECTION FAILED | **Training stopped** |

## When to Use Guard

✅ **USE Guard when:**
- Training on unstable hardware
- Experimenting with aggressive hyperparameters
- Running overnight/weekend jobs unattended
- Cost of restarting is high

❌ **AVOID Guard when:**
- Training is already stable
- You need reproducible results
- Debugging specific training issues
- Production deployments

## The Golden Rule

**Guard is an emergency brake, not a steering wheel.**

It can stop you from crashing, but it won't get you to your destination. Fix the underlying problems - don't rely on Guard to paper over them.

## Validation Checklist

After training with Guard enabled, ALWAYS check:

- [ ] Final eval metrics improved?
- [ ] Training completed faster?
- [ ] Fewer failed runs?
- [ ] Rollback triggers understood?
- [ ] Root causes identified?

If you answered "NO" to any of these, disable Guard and fix the real problems.