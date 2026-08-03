#!/bin/bash
# verify_plot.sh — Generate a plot and verify it with mmx_postool

K=18
CHUNKED=""
PLOT_ID=""
DUMP_PD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --chunked) CHUNKED="--chunked"; shift ;;
        --k) K="$2"; shift 2 ;;
        --dump-pd) DUMP_PD="--dump-pd"; shift ;;
        *) PLOT_ID="$1"; shift ;;
    esac
done

if [ -z "$PLOT_ID" ]; then
    PLOT_ID=$(python3 -c "import os; print(os.urandom(32).hex())")
fi

FARMER_KEY="02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"
RAMDISK="/mnt/ramdisk"

echo "=== K=$K Chunked=${CHUNKED:-flat} PID=${PLOT_ID:0:16}... ==="

rm -f "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot

cd ~/mmx-app/opencl-plotter/build
cp ../f2_f9.cl ../simple_sort.cl ../table_hash.cl . 2>/dev/null

LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter "$PLOT_ID" "$FARMER_KEY" "$RAMDISK" --k $K $CHUNKED $DUMP_PD 2>&1 | grep -E "Done|Built PlotData"

PLOTFILE=$(ls "$RAMDISK"/plot-mmx-hdd-k${K}-*.plot 2>/dev/null | head -1)
if [ -z "$PLOTFILE" ]; then
    echo "ERROR: No plot file generated!"
    exit 1
fi

cd ~/mmx-node
RESULT=$(LD_LIBRARY_PATH=/opt/rocm/lib ./build_opencl/tools/mmx_postool -f "$PLOTFILE" -n 20 -v 2>&1)
echo "$RESULT" | grep -E "Pass:|Fail:|Bad plots:"
