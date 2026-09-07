#!/usr/bin/env bash
set -euo pipefail

: "${ARCHIVE:?Set ARCHIVE to an archive_df_*.pkl checkpoint}"
: "${CONFIG:?Set CONFIG to the experiment cfg.json}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for rendered MP4 files}"

NUM_POLICIES="${NUM_POLICIES:-5}"
EPISODES_PER_POLICY="${EPISODES_PER_POLICY:-1}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.5}"
SEED="${SEED:-20260907}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
FRAME_STRIDE="${FRAME_STRIDE:-2}"

python -m ppga.algorithm.render_mjlab_archive \
  --archive="$ARCHIVE" \
  --config="$CONFIG" \
  --output_dir="$OUTPUT_DIR" \
  --num_policies="$NUM_POLICIES" \
  --episodes_per_policy="$EPISODES_PER_POLICY" \
  --min_success_rate="$MIN_SUCCESS_RATE" \
  --seed="$SEED" \
  --video_width="$VIDEO_WIDTH" \
  --video_height="$VIDEO_HEIGHT" \
  --frame_stride="$FRAME_STRIDE"
