#!/usr/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <input_dir> [output_dir]"
  echo
  echo "  input_dir   Directory with images named 00000.png, 00001.png, …"
  echo "  output_dir  Directory to save output.mp4 (defaults to current dir)"
  exit 1
}

# Check args
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

input_dir=$1
output_dir=${2:-.}

# Validate input directory
if [[ ! -d $input_dir ]]; then
  echo "ERROR: Input directory '$input_dir' does not exist." >&2
  exit 2
fi

# Create output directory if needed
mkdir -p "$output_dir"

# Build output path
output_path="$output_dir/output.mp4"

# Run FFmpeg
ffmpeg -hide_banner -y \
  -start_number 0 -framerate 10 \
  -i "${input_dir}/%05d.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -r 10 -pix_fmt yuv420p \
  "$output_path"

echo "Video saved to $output_path"