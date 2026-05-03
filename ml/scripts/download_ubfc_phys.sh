#!/usr/bin/env bash
# Downloads UBFC-Phys physiological CSVs from IEEE DataPort S3.
# Skips all .avi video files — reduces download from ~780 GB to ~500 MB.
#
# Prerequisites:
#   aws configure   (use your IEEE DataPort AWS credentials)
#
# Usage:
#   bash download_ubfc_phys.sh
#   bash download_ubfc_phys.sh 1 10    # download only subjects 1-10

set -euo pipefail

S3_BASE="s3://ieee-dataport/open/49099/3658"
OUT_DIR="$(dirname "$0")/../data/ubfc_phys"

START=${1:-1}
END=${2:-56}

echo "Downloading UBFC-Phys subjects $START–$END (CSVs/TXT only, no video)"
echo "Output: $OUT_DIR"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

for N in $(seq "$START" "$END"); do
    SUBJECT_DIR="$OUT_DIR/s${N}"
    mkdir -p "$SUBJECT_DIR"

    S3_PATH="${S3_BASE}/s${N}_zip/s${N}"

    FILES=(
        "bvp_s${N}_T1.csv"
        "bvp_s${N}_T2.csv"
        "bvp_s${N}_T3.csv"
        "eda_s${N}_T1.csv"
        "eda_s${N}_T2.csv"
        "eda_s${N}_T3.csv"
        "selfReportedAnx_s${N}.csv"
        "info_s${N}.txt"
    )

    echo "Subject s${N}:"
    SUBJECT_OK=true

    for FILE in "${FILES[@]}"; do
        DEST="$SUBJECT_DIR/$FILE"

        if [[ -f "$DEST" ]]; then
            echo "  ✓ $FILE (already exists)"
            ((SKIPPED++)) || true
            continue
        fi

        if aws s3 cp "${S3_PATH}/${FILE}" "$DEST" --no-progress 2>/dev/null; then
            echo "  ↓ $FILE"
            ((SUCCESS++)) || true
        else
            echo "  ✗ $FILE (failed)"
            ((FAILED++)) || true
            SUBJECT_OK=false
        fi
    done

    if $SUBJECT_OK; then
        echo "  → s${N} complete"
    else
        echo "  → s${N} had errors"
    fi
    echo ""
done

echo "Done."
echo "  Downloaded : $SUCCESS files"
echo "  Skipped    : $SKIPPED files (already existed)"
echo "  Failed     : $FAILED files"
echo ""
echo "Total size on disk:"
du -sh "$OUT_DIR" 2>/dev/null || true
