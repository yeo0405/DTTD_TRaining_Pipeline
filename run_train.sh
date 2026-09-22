export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# PATH CONFIG
# =========================================================

DATASET_ROOT=/home/yeo/Downloads/POSE/dataset
DATASET_CONFIG=/home/yeo/Downloads/POSE/dataset/dataset_config
OUTPUT_DIR=./result_SEP04

# Pretrained model
# PRETRAIN="./result_small_AUG19_m2p1_0821-xxxx/checkpoints/best.pth"


# =========================================================
# MODEL CONFIG
# =========================================================

BASE_LATENT=128
EMBED_DIM=256
FUSION_BLOCK_NUM=1
LAYER_NUM_M=2
LAYER_NUM_P=1


# =========================================================
# TRAINING CONFIG
# =========================================================

RECON_W=0.3
RECON_CHOICE=depth
LOSS=adds
OPTIM_BATCH=4
START_EPOCH=0

LR=1e-5
MIN_LR=1e-6
LR_RATE=0.3

DECAY_MARGIN=0.033
DECAY_RATE=0.77

NEPOCH=30
WARM_EPOCH=1


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
# Check Pretrain Model
# =========================================================

if [ -n "$PRETRAIN" ]; then

    if [ ! -f "$PRETRAIN" ]; then
        log_error "Pretrain model not found: $PRETRAIN"
    fi

    log_info "Using pretrained model:"
    log_info "$PRETRAIN"

else

    log_info "No pretrained model specified."
    log_info "Training from scratch."

fi


# =========================================================
# Step 1: Train the model
# =========================================================

python train.py \
    --device 0 \
    --dataset_root "$DATASET_ROOT" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --base_latent "$BASE_LATENT" \
    --embed_dim "$EMBED_DIM" \
    --fusion_block_num "$FUSION_BLOCK_NUM" \
    --layer_num_m "$LAYER_NUM_M" \
    --layer_num_p "$LAYER_NUM_P" \
    --recon_w "$RECON_W" \
    --recon_choice "$RECON_CHOICE" \
    --loss "$LOSS" \
    --optim_batch "$OPTIM_BATCH" \
    --start_epoch "$START_EPOCH" \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --lr_rate "$LR_RATE" \
    --decay_margin "$DECAY_MARGIN" \
    --decay_rate "$DECAY_RATE" \
    --nepoch "$NEPOCH" \
    --warm_epoch "$WARM_EPOCH" \
    --filter_enhance \
    # ${PRETRAIN:+--pretrain "$PRETRAIN"}