#!/bin/bash
set -e

# ============================================================
# Configuration
# ============================================================

PYTHON="python"

INPUT_DIR="/home/yeo/Downloads/FoundationPose/dataset/iphone_14_pro_max_rotation"
OBJECTS_SOURCE_PATH="/home/yeo/Downloads/DTTD_obj/iphone_with_box"
OUTPUT_DIR="/home/yeo/Downloads/POSE/iphone_14_pro_max_rotation"

OBJ_ID=4

SPLIT_RATIO="0.8"
SEED="42"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================
# Logging
# ============================================================

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

# ============================================================
# Validation
# ============================================================

log_info "Validating configuration..."

if [ ! -d "$INPUT_DIR" ]; then
    log_error "INPUT_DIR does not exist: $INPUT_DIR"
fi

if [ ! -d "$OBJECTS_SOURCE_PATH" ]; then
    log_error "OBJECTS_SOURCE_PATH does not exist: $OBJECTS_SOURCE_PATH"
fi

if [ ! -f "$SCRIPT_DIR/foundation_pose_to_training.py" ]; then
    log_error "foundation_pose_to_training.py not found: $SCRIPT_DIR/foundation_pose_to_training.py"
fi

if [ ! -f "$SCRIPT_DIR/val_split.py" ]; then
    log_error "val_split.py not found: $SCRIPT_DIR/val_split.py"
fi

log_success "Configuration validated"

# ============================================================
# Print Configuration
# ============================================================

log_info ""
log_info "========================================================="
log_info "Data Preprocessing Configuration"
log_info "========================================================="
log_info "PYTHON              : $PYTHON"
log_info "INPUT_DIR           : $INPUT_DIR"
log_info "OBJECTS_SOURCE_PATH : $OBJECTS_SOURCE_PATH"
log_info "OUTPUT_DIR          : $OUTPUT_DIR"
log_info "OBJ_ID              : $OBJ_ID"
log_info "SPLIT_RATIO         : $SPLIT_RATIO"
log_info "SEED                : $SEED"
log_info "SCRIPT_DIR          : $SCRIPT_DIR"
log_info "========================================================="
log_info ""

# ============================================================
# Step 0: Copy objects folder
# ============================================================

log_info "Step 0: Copying objects folder..."

mkdir -p "$OUTPUT_DIR/objects"

cp -a "$OBJECTS_SOURCE_PATH"/. \
      "$OUTPUT_DIR/objects"/

log_success "Step 0 completed"

# ============================================================
# Step 1: Generate objectids.csv
# ============================================================

log_info "Step 1: Generating objectids.csv..."

OBJECTIDS_CSV="$OUTPUT_DIR/dataset_config/objectids.csv"

mkdir -p "$OUTPUT_DIR/dataset_config"

{
    echo "id,name,symmetry"

    idx=1

    while IFS= read -r obj_dir; do
        name="$(basename "$obj_dir")"
        symmetry=0

        echo "${idx},${name},${symmetry}"

        idx=$((idx + 1))
    done < <(
        find "$OBJECTS_SOURCE_PATH" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            | sort
    )

} > "$OBJECTIDS_CSV"

log_success "Step 1 completed: $OBJECTIDS_CSV"

# ============================================================
# Step 2: Convert FoundationPose -> DTTD
# ============================================================

log_info "Step 2: Converting FoundationPose format to DTTD format..."

$PYTHON "$SCRIPT_DIR/foundation_pose_to_training.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --obj_id "$OBJ_ID"

log_success "Step 2 completed"

# ============================================================
# Step 3: Split train / val
# ============================================================

log_info "Step 3: Splitting data into train and val..."

$PYTHON "$SCRIPT_DIR/val_split.py" \
    --converted_root "$OUTPUT_DIR" \
    --split_ratio "$SPLIT_RATIO" \
    --seed "$SEED"

log_success "Step 3 completed"

# ============================================================
# Summary
# ============================================================

log_info ""
log_success "Data preprocessing completed successfully!"

log_info "Output directory: $OUTPUT_DIR"

log_info ""
log_info "Dataset structure:"
log_info "  $OUTPUT_DIR/data/"
log_info "    - Processed image data"
log_info ""
log_info "  $OUTPUT_DIR/dataset_config/"
log_info "    - objectids.csv"
log_info ""
log_info "  $OUTPUT_DIR/objects/"
log_info "    - Object files and folders"
log_info ""
log_info "  $OUTPUT_DIR/train_data_list.txt"
log_info "    - Training data list"
log_info ""
log_info "  $OUTPUT_DIR/test_data_list.txt"
log_info "    - Test/validation data list"