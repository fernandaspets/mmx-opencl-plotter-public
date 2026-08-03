#!/bin/bash
# Screen PIDs: test if k18 proof count predicts k25 proof count
# Uses match count (final_Y) as fast proxy + postool for actual proofs
export LD_LIBRARY_PATH=/opt/rocm/lib
PLOTTER=~/mmx-app/opencl-plotter/build/mmx_opencl_plotter
POSTOOL=~/mmx-node/build/tools/mmx_postool
FK="02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"
OUTDIR=/dev/shm/plotram
DERIVE=~/mmx-app/opencl-plotter/derive_pid.py

N=${1:-10}
RESULTS=~/mmx-app/opencl-plotter/pid_screening_results.csv

echo "seed,k18_pid,k18_finalY,k18_proofs,k18_time,k25_pid,k25_finalY,k25_proofs,k25_time" > $RESULTS

echo "Generating $N seeds..."
SEEDS=$(python3 $DERIVE $N)

count=0
for line in $SEEDS; do
    count=$((count + 1))
    SEED=$(echo $line | cut -d, -f1)
    K18_PID=$(echo $line | cut -d, -f2)
    K25_PID=$(echo $line | cut -d, -f3)
    
    # --- Run k18 ---
    rm -f $OUTDIR/*.plot
    K18_START=$(date +%s.%N)
    K18_OUTPUT=$($PLOTTER "$K18_PID" "$FK" $OUTDIR/ --k 18 --no-yield 2>&1)
    K18_END=$(date +%s.%N)
    K18_TIME=$(echo "$K18_END - $K18_START" | bc)
    K18_FY=$(echo "$K18_OUTPUT" | grep "final_Y=" | grep -oP '\d+(?=\s)' | head -1)
    K18_PROOFS="crash"
    PLOTFILE=$(ls $OUTDIR/*.plot 2>/dev/null | head -1)
    if [ -n "$PLOTFILE" ]; then
        K18_PROOFS=$($POSTOOL -f "$PLOTFILE" -n 18 -v 2>&1 | grep "Pass:" | grep -oP '\d+(?= /)' | head -1)
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
    
    [ -z "$K18_FY" ] && K18_FY=0
    [ -z "$K18_PROOFS" ] && K18_PROOFS="crash"
    [ -z "$K25_FY" ] && K25_FY=0
    [ -z "$K25_PROOFS" ] && K25_PROOFS="crash"
    
    echo "$SEED,$K18_PID,$K18_FY,$K18_PROOFS,$K18_TIME,$K25_PID,$K25_FY,$K25_PROOFS,$K25_TIME" >> $RESULTS
    echo "  [$count/$N] k18: FY=$K18_FY proofs=$K18_PROOFS (${K18_TIME}s) | k25: FY=$K25_FY proofs=$K25_PROOFS (${K25_TIME}s)"
done

echo ""
echo "=== Results ==="
column -t -s, $RESULTS
echo ""

# Compute correlation
echo "=== Correlation Analysis ==="
python3 << 'PYEOF'
import csv
import sys

with open("~/mmx-app/opencl-plotter/pid_screening_results.csv".replace("~", "/home/ubman")) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter out crashes
valid = [r for r in rows if r["k18_proofs"] != "crash" and r["k25_proofs"] != "crash" 
         and r["k18_proofs"] and r["k25_proofs"]]

if len(valid) < 3:
    print(f"Not enough valid data points ({len(valid)}). Need at least 3.")
    sys.exit(0)

k18_proofs = [int(r["k18_proofs"]) for r in valid]
k25_proofs = [int(r["k25_proofs"]) for r in valid]
k18_fy = [int(r["k18_finalY"]) for r in valid]
k25_fy = [int(r["k25_finalY"]) for r in valid]

# Sort by k18 proofs
sorted_data = sorted(zip(k18_proofs, k25_proofs, k18_fy, k25_fy))

print(f"Valid samples: {len(valid)}")
print(f"k18 proofs: min={min(k18_proofs)}, max={max(k18_proofs)}, avg={sum(k18_proofs)/len(k18_proofs):.1f}")
print(f"k25 proofs: min={min(k25_proofs)}, max={max(k25_proofs)}, avg={sum(k25_proofs)/len(k25_proofs):.1f}")
print(f"k18 final_Y: min={min(k18_fy)}, max={max(k18_fy)}, avg={sum(k18_fy)/len(k18_fy):.1f}")
print(f"k25 final_Y: min={min(k25_fy)}, max={max(k25_fy)}, avg={sum(k25_fy)/len(k25_fy):.1f}")

# Simple Pearson correlation
n = len(valid)
mean_x = sum(k18_proofs) / n
mean_y = sum(k25_proofs) / n
cov = sum((k18_proofs[i] - mean_x) * (k25_proofs[i] - mean_y) for i in range(n))
var_x = sum((k18_proofs[i] - mean_x) ** 2 for i in range(n))
var_y = sum((k25_proofs[i] - mean_y) ** 2 for i in range(n))
if var_x > 0 and var_y > 0:
    corr = cov / (var_x ** 0.5 * var_y ** 0.5)
    print(f"\nPearson correlation (k18 proofs vs k25 proofs): {corr:.4f}")
else:
    print(f"\nCannot compute correlation (zero variance)")

# Correlation for final_Y
mean_fx = sum(k18_fy) / n
mean_fy = sum(k25_fy) / n
cov_f = sum((k18_fy[i] - mean_fx) * (k25_fy[i] - mean_fy) for i in range(n))
var_fx = sum((k18_fy[i] - mean_fx) ** 2 for i in range(n))
var_fy = sum((k25_fy[i] - mean_fy) ** 2 for i in range(n))
if var_fx > 0 and var_fy > 0:
    corr_f = cov_f / (var_fx ** 0.5 * var_fy ** 0.5)
    print(f"Pearson correlation (k18 final_Y vs k25 final_Y): {corr_f:.4f}")

# Show top and bottom k18 performers
print(f"\n=== Sorted by k18 proofs (best to worst) ===")
for k18p, k25p, k18f, k25f in sorted_data:
    print(f"  k18={k18p:4d} proofs (FY={k18f:6d}) → k25={k25p:4d} proofs (FY={k25f:8d})")

# Top half vs bottom half
half = len(sorted_data) // 2
if half > 0:
    top_k25 = sum(k25 for _, k25, _, _ in sorted_data[:half]) / half
    bot_k25 = sum(k25 for _, k25, _, _ in sorted_data[half:]) / max(len(sorted_data) - half, 1)
    print(f"\n=== Top half (by k18) vs Bottom half (by k18) ===")
    print(f"  Top half k25 avg proofs:    {top_k25:.1f}")
    print(f"  Bottom half k25 avg proofs: {bot_k25:.1f}")
    print(f"  Ratio: {top_k25/bot_k25:.4f}x" if bot_k25 > 0 else "  N/A")
PYEOF
