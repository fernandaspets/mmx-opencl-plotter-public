#!/bin/bash
# Screen PIDs: test if k22 proof count predicts k25 proof count
export LD_LIBRARY_PATH=/opt/rocm/lib
PLOTTER=~/mmx-app/opencl-plotter/build/mmx_opencl_plotter
POSTOOL=~/mmx-node/build/tools/mmx_postool
FK="02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"
OUTDIR=/dev/shm/plotram
DERIVE=~/mmx-app/opencl-plotter/derive_pid.py

N=${1:-10}
RESULTS=~/mmx-app/opencl-plotter/pid_screening_k22_results.csv

echo "seed,k22_pid,k22_finalY,k22_proofs,k22_time,k25_pid,k25_finalY,k25_proofs,k25_time" > $RESULTS

echo "Generating $N seeds..."
SEEDS=$(python3 $DERIVE $N)

count=0
for line in $SEEDS; do
    count=$((count + 1))
    SEED=$(echo $line | cut -d, -f1)
    K22_PID=$(echo $line | cut -d, -f2)
    K25_PID=$(echo $line | cut -d, -f3)
    
    # --- Run k22 ---
    rm -f $OUTDIR/*.plot
    K22_START=$(date +%s.%N)
    K22_OUTPUT=$($PLOTTER "$K22_PID" "$FK" $OUTDIR/ --k 22 --no-yield 2>&1)
    K22_END=$(date +%s.%N)
    K22_TIME=$(echo "$K22_END - $K22_START" | bc)
    K22_FY=$(echo "$K22_OUTPUT" | grep "final_Y=" | grep -oP '\d+(?=\s)' | head -1)
    K22_PROOFS="crash"
    PLOTFILE=$(ls $OUTDIR/*.plot 2>/dev/null | head -1)
    if [ -n "$PLOTFILE" ]; then
        K22_PROOFS=$($POSTOOL -f "$PLOTFILE" -n 22 -v 2>&1 | grep "Pass:" | grep -oP '\d+(?= /)' | head -1)
    fi
    
    # --- Run k25 ---
    rm -f $OUTDIR/*.plot
    K25_START=$(date +%s.%N)
    K25_OUTPUT=$($PLOTTER "$K25_PID" "$FK" $OUTDIR/ --k 25 --no-yield 2>&1)
    K25_END=$(date +%s.%N)
    K25_TIME=$(echo "$K25_END - $K25_START" | bc)
    K25_FY=$(echo "$K25_OUTPUT" | grep "final_Y=" | grep -oP '\d+(?=\s)' | head -1)
    K25_PROOFS="crash"
    PLOTFILE=$(ls $OUTDIR/*.plot 2>/dev/null | head -1)
    if [ -n "$PLOTFILE" ]; then
        K25_PROOFS=$($POSTOOL -f "$PLOTFILE" -n 25 -v 2>&1 | grep "Pass:" | grep -oP '\d+(?= /)' | head -1)
    fi
    
    [ -z "$K22_FY" ] && K22_FY=0
    [ -z "$K22_PROOFS" ] && K22_PROOFS="crash"
    [ -z "$K25_FY" ] && K25_FY=0
    [ -z "$K25_PROOFS" ] && K25_PROOFS="crash"
    
    echo "$SEED,$K22_PID,$K22_FY,$K22_PROOFS,$K22_TIME,$K25_PID,$K25_FY,$K25_PROOFS,$K25_TIME" >> $RESULTS
    echo "  [$count/$N] k22: FY=$K22_FY proofs=$K22_PROOFS (${K22_TIME}s) | k25: FY=$K25_FY proofs=$K25_PROOFS (${K25_TIME}s)"
done

echo ""
echo "=== Correlation Analysis ==="
python3 << 'PYEOF'
import csv, sys

with open("/home/ubman/mmx-app/opencl-plotter/pid_screening_k22_results.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

valid = [r for r in rows if r["k22_proofs"] not in ("crash", "") and r["k25_proofs"] not in ("crash", "")
         and r["k22_proofs"] and r["k25_proofs"]]

if len(valid) < 3:
    print(f"Not enough valid data points ({len(valid)}). Need at least 3.")
    sys.exit(0)

k22_proofs = [int(r["k22_proofs"]) for r in valid]
k25_proofs = [int(r["k25_proofs"]) for r in valid]
k22_fy = [int(r["k22_finalY"]) for r in valid]
k25_fy = [int(r["k25_finalY"]) for r in valid]

n = len(valid)
mean_x = sum(k22_proofs) / n
mean_y = sum(k25_proofs) / n
cov = sum((k22_proofs[i] - mean_x) * (k25_proofs[i] - mean_y) for i in range(n))
var_x = sum((k22_proofs[i] - mean_x) ** 2 for i in range(n))
var_y = sum((k25_proofs[i] - mean_y) ** 2 for i in range(n))

print(f"Valid samples: {len(valid)}")
print(f"k22 proofs: min={min(k22_proofs)}, max={max(k22_proofs)}, avg={mean_x:.1f}, range={max(k22_proofs)-min(k22_proofs)}")
print(f"k25 proofs: min={min(k25_proofs)}, max={max(k25_proofs)}, avg={mean_y:.1f}, range={max(k25_proofs)-min(k25_proofs)}")
print(f"k22 final_Y: min={min(k22_fy)}, max={max(k22_fy)}, avg={sum(k22_fy)/n:.1f}")
print(f"k25 final_Y: min={min(k25_fy)}, max={max(k25_fy)}, avg={sum(k25_fy)/n:.1f}")

if var_x > 0 and var_y > 0:
    corr = cov / (var_x ** 0.5 * var_y ** 0.5)
    print(f"\n*** Pearson correlation (k22 proofs vs k25 proofs): {corr:.4f} ***")
    if abs(corr) < 0.3:
        print("    → NO correlation. Screening at k22 would NOT help pick good k25 PIDs.")
    elif corr > 0.5:
        print("    → STRONG positive correlation. Screening at k22 WOULD help!")
    elif corr < -0.5:
        print("    → STRONG negative correlation. Best k22 = worst k25!")
    else:
        print("    → WEAK correlation. Screening might help marginally.")

# Correlation for final_Y
mean_fx = sum(k22_fy) / n
mean_fy = sum(k25_fy) / n
cov_f = sum((k22_fy[i] - mean_fx) * (k25_fy[i] - mean_fy) for i in range(n))
var_fx = sum((k22_fy[i] - mean_fx) ** 2 for i in range(n))
var_fy = sum((k25_fy[i] - mean_fy) ** 2 for i in range(n))
if var_fx > 0 and var_fy > 0:
    corr_f = cov_f / (var_fx ** 0.5 * var_fy ** 0.5)
    print(f"*** Pearson correlation (k22 final_Y vs k25 final_Y): {corr_f:.4f} ***")

# Sort and display
sorted_data = sorted(zip(k22_proofs, k25_proofs, k22_fy, k25_fy))
print(f"\n=== Sorted by k22 proofs (best to worst) ===")
for k22p, k25p, k22f, k25f in sorted_data:
    marker = " ★" if k25p == max(k25_proofs) else ""
    print(f"  k22={k22p:4d} proofs (FY={k22f:8d}) → k25={k25p:4d} proofs (FY={k25f:8d}){marker}")

half = len(sorted_data) // 2
if half > 0 and len(sorted_data) > half:
    top_k25 = sum(k25 for _, k25, _, _ in sorted_data[:half]) / half
    bot_k25 = sum(k25 for _, k25, _, _ in sorted_data[half:]) / max(len(sorted_data) - half, 1)
    print(f"\n=== Top half (by k22) vs Bottom half (by k22) ===")
    print(f"  Top half k25 avg proofs:    {top_k25:.1f}")
    print(f"  Bottom half k25 avg proofs: {bot_k25:.1f}")
    if bot_k25 > 0:
        print(f"  Ratio: {top_k25/bot_k25:.4f}x (>1.0 means screening helps)")
PYEOF
