#!/usr/bin/env python3
"""
test_pd_chain.py — Verify PD chain integrity for a plot file.

Reads PD dump files (/tmp/pd_flat_T*.txt or /tmp/pd_chunked_T*.txt)
and verifies that the PD chain is consistent:
  PD[t][i].pos must be a valid index into PD[t-1]
  PD[t][i].pos + PD[t][i].delta must also be a valid index into PD[t-1]

Usage: python3 test_pd_chain.py flat|chunked
"""

import sys
import re
import os

def parse_pd_dump(prefix):
    """Parse PD dump files, return dict {table: [(pos, delta), ...]}"""
    tables = {}
    for t in range(2, 10):
        fname = f"/tmp/pd_{prefix}_T{t}.txt"
        if not os.path.exists(fname):
            continue
        entries = []
        with open(fname) as f:
            for line in f:
                m = re.search(r'PD\[\d+\]\[\d+\] = \(pos=(\d+), delta=(\d+)\)', line)
                if m:
                    entries.append((int(m.group(1)), int(m.group(2))))
        if entries:
            tables[t] = entries
    return tables

def verify_chain(tables):
    """Verify PD chain: PD[t][i].pos and .pos+.delta must be valid in PD[t-1]"""
    errors = []
    for t in sorted(tables.keys()):
        if t == 2:
            continue  # PD[2] positions index into X table, not PD[1]
        if t - 1 not in tables:
            errors.append(f"T{t}: PD[{t-1}] not available for verification")
            continue
        
        prev_size = len(tables[t-1])
        # Note: we only have the first 50 entries from the dump
        # So we check only the first 50
        for i, (pos, delta) in enumerate(tables[t][:50]):
            # pos must be a valid index into PD[t-1]
            if pos >= prev_size:
                errors.append(f"T{t}[{i}]: pos={pos} >= PD[{t-1}].size={prev_size}")
                continue
            right_pos = pos + delta
            if right_pos >= prev_size:
                errors.append(f"T{t}[{i}]: pos+delta={right_pos} >= PD[{t-1}].size={prev_size}")
    
    return errors

def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "flat"
    tables = parse_pd_dump(prefix)
    
    print(f"=== PD Chain Verification ({prefix}) ===")
    for t in sorted(tables.keys()):
        print(f"  PD[{t}]: {len(tables[t])} entries (dumped)")
    
    errors = verify_chain(tables)
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
    else:
        print(f"\n  PD chain OK (first 50 entries per table)")
    
    # Also check final_Y
    fy_file = f"/tmp/pd_{prefix}_finalY.txt"
    if os.path.exists(fy_file):
        ys = []
        with open(fy_file) as f:
            for line in f:
                m = re.search(r'Y\[\d+\] = (\d+)', line)
                if m:
                    ys.append(int(m.group(1)))
        if ys:
            print(f"\n  final_Y: {len(ys)} entries (dumped)")
            print(f"  Y[0..4]: {ys[:5]}")
            # Check if sorted
            is_sorted = all(ys[i] <= ys[i+1] for i in range(len(ys)-1))
            print(f"  Sorted: {is_sorted}")

if __name__ == "__main__":
    main()
