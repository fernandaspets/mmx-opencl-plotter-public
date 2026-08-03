#!/bin/bash
# compare.sh — Run flat and chunked with same plot_id, compare results
# Usage: ./tools/compare.sh [--k 18]

K=18
if [ "$1" == "--k" ]; then K="$2"; fi

PLOT_ID=$(python3 -c "import os; print(os.urandom(32).hex())")
FARMER_KEY="02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"
RAMDISK="/mnt/ramdisk"

echo "=== Comparing Flat vs Chunked (K=$K, PID=${PLOT_ID:0:16}...) ==="
echo ""

# FLAT
echo "--- FLAT ---"
rm -f "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot /tmp/pd_flat_*.txt
cd ~/mmx-app/opencl-plotter/build
cp ../f2_f9.cl ../simple_sort.cl ../table_hash.cl . 2>/dev/null
LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter "$PLOT_ID" "$FARMER_KEY" "$RAMDISK" --k $K --dump-pd 2>&1 | grep -E "Done"
PLOTFILE=$(ls "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot 2>/dev/null | head -1)
cd ~/mmx-node
LD_LIBRARY_PATH=/opt/rocm/lib ./build_opencl/tools/mmx_postool -f "$PLOTFILE" -n 20 -v 2>&1 | grep -E "Pass:|Bad"

echo ""

# CHUNKED
echo "--- CHUNKED ---"
rm -f "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot /tmp/pd_chunked_*.txt
cd ~/mmx-app/opencl-plotter/build
nohup bash -c "LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter \"$PLOT_ID\" \"$FARMER_KEY\" \"$RAMDISK\" --k $K --chunked --dump-pd > /tmp/chunked_compare.log 2>&1" &
# Wait for completion
while kill -0 $! 2>/dev/null; do sleep 1; done
grep -E "Done|Built PlotData" /tmp/chunked_compare.log
PLOTFILE=$(ls "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot 2>/dev/null | head -1)
if [ -n "$PLOTFILE" ]; then
    cd ~/mmx-node
    LD_LIBRARY_PATH=/opt/rocm/lib ./build_opencl/tools/mmx_postool -f "$PLOTFILE" -n 20 -v 2>&1 | grep -E "Pass:|Bad"
else
    echo "No plot file (crashed?)"
fi

echo ""
echo "=== PD Comparison ==="
for t in 2 3 9; do
    echo "T$t flat:   $(grep "PD\[$t\]\[0\]" /tmp/pd_flat_T$t.txt 2>/dev/null | head -1)"
    echo "T$t chunked: $(grep "PD\[$t\]\[0\]" /tmp/pd_chunked_T$t.txt 2>/dev/null | head -1)"
done
echo ""
echo "final_Y flat:   $(grep "Y\[0\]" /tmp/pd_flat_finalY.txt 2>/dev/null)"
echo "final_Y chunked: $(grep "Y\[0\]" /tmp/pd_chunked_finalY.txt 2>/dev/null)"
