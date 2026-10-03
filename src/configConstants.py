######################################################################
# CONFIGURATION - Global Constants and Keys
######################################################################
"""
Central configuration file for the scribble segmentation project.

This module defines all global constants, dictionary keys, and configuration
values used throughout the project. Avoid hardcoding these values elsewhere.

Constants Categories:
    - Scribble types and generation methods
    - Dataset keys for tensor dictionaries
    - Color identifiers for visualization
    - Folder and file naming conventions
    - Training configuration keys
"""

######################################################################
# SCRIBBLE TYPES
######################################################################

SCRIBBLE_TYPE_HUMAN    = 'human'
SCRIBBLE_TYPE_CONTOUR  = 'contour'
SCRIBBLE_TYPE_SKELETON = 'skeleton'
SCRIBBLE_TYPE_MEDIAL   = 'medial'

######################################################################
# SCRIBBLE CLASS TYPES (Foreground vs Background)
######################################################################

# Numeric class convention used everywhere (dataset, model embedding, eval, inference):
#   0 = background scribble (drawn on FP: prediction=1, GT=0)
#   1 = foreground scribble (drawn on GT: FN ∪ TP)
SCRIBBLE_CLASS_FOREGROUND = 'foreground'  # Scribbles on GT regions (value=1)
SCRIBBLE_CLASS_BACKGROUND = 'background'  # Scribbles on FP regions (value=0)
SCRIBBLE_CLASS_BACKGROUND_VALUE = 0
SCRIBBLE_CLASS_FOREGROUND_VALUE = 1

######################################################################
# METADATA KEYS
######################################################################

KEY_SCRIBBLE_GENERATION_TYPE = "scribble_generation_type"
KEY_SCRIBBLE_PATH            = "scribble_data_type"
KEY_TRANSFORMS               = "transforms"
KEY_DATA_SOURCE              = "data_source"

DATA_SOURCE_REAL             = "real"
DATA_SOURCE_SYNTHETIC_PERLIN = "synthetic_perlin"
DATA_SOURCE_SYNTHETIC_SHAPE  = "synthetic_shape"

######################################################################
# SHAPE CONFIGURATION
######################################################################

KEY_SHAPE = "shape"
SHAPE_VALUE_ANY = "any"

######################################################################
# DATASET RETURN KEYS
######################################################################

KEY_GT                 = "gt"
KEY_PREDICTION         = "prediction"
KEY_SCRIBBLE_TYPE      = "scribble_type"
KEY_SCRIBBLE_CLASS     = "scribble_class"  # 'foreground', 'background', or 'both'
KEY_SCRIBBLE           = "scribble"
KEY_SCRIBBLE_GAUSS     = "scribble_gauss"
KEY_META               = "meta"

######################################################################
# VISUALIZATION COLORS
######################################################################

KEY_COLOR_GREEN  = "green"
KEY_COLOR_RED    = "red"
KEY_COLOR_BLUE   = "blue"
KEY_COLOR_YELLOW = "yellow"

######################################################################
# FOLDER AND FILE NAMES
######################################################################

FOLDERNAME_TMP = "_tmp"

######################################################################
# TRANSFORM KEYS
######################################################################

KEY_ROTATE   = "rotate"
KEY_FLIP_HOR = "flip_hor"
KEY_FLIP_VER = "flip_ver"
KEY_TRANS    = "trans"

######################################################################
# MODEL FILE NAMES
######################################################################

FILENAME_MODEL = "model.bin"
FILENAME_OPTIMIZER = "optimizer.pt"
FILENAME_SCHEDULER = "scheduler.pt"

######################################################################
# TRAINING CONFIGURATION KEYS
######################################################################

KEY_CLIP_NORM = "clip_norm"
KEY_LOG_EVERY = "log_every"
KEY_NUM_EPOCHS = "num_epochs"
KEY_SEED = "seed"
KEY_OPTIMIZER = "optimizer"
KEY_WANDB_PROJECT = "wandb_project"
KEY_WANDB_NAME = "wandb_name"
KEY_WANDB_NOTES = "wandb_notes"

######################################################################
# TRAINING MODES
######################################################################

KEY_TRAIN = "train"
KEY_VAL   = "val"

######################################################################
# DATASET THRESHOLDS AND DEFAULTS
######################################################################

MIN_SUM_SCRIBBLE = 1
DEFAULT_SEED_VALUE = 42
DEFAULT_LOG_EVERY = 10

######################################################################
# MODEL CONFIGURATION
######################################################################

KEY_CONFIG_MODEL = "model"
KEY_ADAM = "Adam"
KEY_ADAMW = "AdamW"
LOSS_L1 = "L1"
LOSS_L2 = "L2"

MODEL_DU_UNET_4BLOCKS = "duUnet4Blocks"

######################################################################
# SYNTHETIC DATA CONFIGURATION
######################################################################

KEY_SYNTHETIC_PERLIN_GENERATED_PROBABILITY = "synthetic_perlin_generated_probability"
MED_SCRIBBLE_PATH = "med_scribble_path"

######################################################################
# PRETRAINING CONFIGURATION

######################################################################
# MODEL CONDITIONAL CONFIGURATION
######################################################################

MODEL_CONDITIONAL = "conditional"
KEY_CONDITIONAL_LOSS = "conditional_loss"