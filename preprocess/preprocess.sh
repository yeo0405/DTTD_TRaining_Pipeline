#!/bin/bash
set -e

# =========================================================
# Data Preprocessing Configuration
# =========================================================

BOP_DATA_PATH="${BOP_DATA_PATH:-/home/yeo/Downloads/Data_Generation/base/bop_data/hb}"
OBJECTS_SOURCE_PATH="${OBJECTS_SOURCE_PATH:-/home/yeo/Downloads/POSE/objects}"
OUTPUT_DIR="${OUTPUT_DIR:-../dataset}"
OBJ_ID="${OBJ_ID:-0}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# =========================================================
# Logging Functions
# =========================================================

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
    exit 1
}

log_success() {
    echo "[SUCCESS] $1"
}

# =========================================================
# Validation
# =========================================================

log_info "Validating configuration..."

if [ ! -d "$BOP_DATA_PATH" ]; then
    log_error "BOP_DATA_PATH does not exist: $BOP_DATA_PATH"
fi

if [ ! -d "$OBJECTS_SOURCE_PATH" ]; then
    log_error "OBJECTS_SOURCE_PATH does not exist: $OBJECTS_SOURCE_PATH"
fi

if [ ! -d "$SCRIPT_DIR" ]; then
    log_error "SCRIPT_DIR does not exist: $SCRIPT_DIR"
fi

if [ ! -f "$SCRIPT_DIR/transfer.py" ]; then
    log_error "transfer.py not found in SCRIPT_DIR: $SCRIPT_DIR"
fi

if [ ! -f "$SCRIPT_DIR/val_split.py" ]; then
    log_error "val_split.py not found in SCRIPT_DIR: $SCRIPT_DIR"
fi

if [ ! -f "$SCRIPT_DIR/augment.py" ]; then
    log_error "augment.py not found in SCRIPT_DIR: $SCRIPT_DIR"
fi

log_success "Configuration validated"

# =========================================================
# Step 0: Copy objects folder
# =========================================================

log_info "Step 0: Copying objects folder..."
log_info "  OBJECTS_SOURCE_PATH: $OBJECTS_SOURCE_PATH"
log_info "  OUTPUT_DIR: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/objects"
cp -a "$OBJECTS_SOURCE_PATH"/. "$OUTPUT_DIR/objects"/

log_success "Step 0 completed"

# =========================================================
# Step 1: Generate objectids.csv
# =========================================================

log_info "Step 1: Generating objectids.csv..."

OBJECTS_DIR="$OBJECTS_SOURCE_PATH"
OBJECTIDS_CSV="$OUTPUT_DIR/dataset_config/objectids.csv"

if [ ! -d "$OBJECTS_DIR" ]; then
    log_error "OBJECTS_DIR does not exist: $OBJECTS_DIR"
fi

mkdir -p "$OUTPUT_DIR/dataset_config"

{
    echo "id,name,symmetry"

    idx=1

    while IFS= read -r obj_dir; do
        name="$(basename "$obj_dir")"

        # default: non-symmetric
        symmetry=0

        echo "${idx},${name},${symmetry}"

        idx=$((idx + 1))

    done < <(find "$OBJECTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

} > "$OBJECTIDS_CSV"

log_success "Step 1 completed: $OBJECTIDS_CSV"

# =========================================================
# Step 2: Convert BOP format to DTTD format
# =========================================================

log_info "Step 2: Converting BOP format to DTTD format..."
log_info "  BOP_DATA_PATH: $BOP_DATA_PATH"
log_info "  OUTPUT_DIR: $OUTPUT_DIR"
log_info "  OBJ_ID: $OBJ_ID"

$PYTHON "$SCRIPT_DIR/transfer_v2.py" \
    --bop_data_path "$BOP_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --obj_id "$OBJ_ID"

log_success "Step 2 completed"

# =========================================================
# Step 3: Split data into train and val
# =========================================================

log_info "Step 3: Splitting data into train and val..."
log_info "  SPLIT_RATIO: $SPLIT_RATIO"
log_info "  SEED: $SEED"

$PYTHON "$SCRIPT_DIR/val_split.py" \
    --converted_root "$OUTPUT_DIR" \
    --split_ratio "$SPLIT_RATIO" \
    --seed "$SEED"

log_success "Step 3 completed"

# =========================================================
# Step 4: Augment training data
# =========================================================

# log_info "Step 4: Augmenting training data..."

# $PYTHON "$SCRIPT_DIR/augment_depth.py" \
#     --dataset_dir "$OUTPUT_DIR"

# log_success "Step 4 completed"

# =========================================================
# Summary
# =========================================================

log_info ""
log_success "Data preprocessing completed successfully!"
log_info "Output directory: $OUTPUT_DIR"
log_info ""
log_info "Dataset structure:"
log_info "  $OUTPUT_DIR/data/           - Processed image data"
log_info "  $OUTPUT_DIR/dataset_config/  - Configuration files"
log_info "  $OUTPUT_DIR/objects/         - Object files and folders"
log_info "    - train_data_list.txt      - Training data list"
log_info "    - test_data_list.txt       - Test/validation data list"