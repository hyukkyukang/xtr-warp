import torch

# Store tokenizer names/paths instead of the actual tokenizers
SRC_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
TGT_TOKENIZER_NAME = "google/xtr-base-en"

CHUNK_LENGTH = 64
INPUT_LENGTH = 1024

INDEX_ROOT = "/root/warp/indexes"
COLLECTION_PATH = "/mnt/md0/hkkang/retro/data/huggingface/pints_ai/meta-llama_Llama-3.2-1B_tokenized/segment_cache_1024_0"
DATASET_NAME = "pints_ai"

NBITS = 4
K = 10
NRANKS = torch.cuda.device_count()
BATCH_SIZE = 8
NUM_WORKERS = 2
RESUME = False
