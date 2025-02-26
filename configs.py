import torch

# Store tokenizer names/paths instead of the actual tokenizers
SRC_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
TGT_TOKENIZER_NAME = "google/xtr-base-en"

CHUNK_LENGTH = 64
INPUT_LENGTH = 1024

INDEX_ROOT = "/root/warp/indexes"
INDEX_PATH = "/root/warp/collections/"
COLLECTION_PATH = "/mnt/md0/hkkang/retro/data/huggingface/pints_ai/meta-llama_Llama-3.2-1B_tokenized/segment_cache_1024_0"
DATASET_NAME = "pints_ai_full"

NBITS = 4
K = 10
NRANKS = torch.cuda.device_count()
BATCH_SIZE = 16
NUM_WORKERS = 2
RESUME = False

# This is the rate of the collection that will be used
# COLLECTION_SAMPLE_RATE = 0.01
COLLECTION_SAMPLE_RATE = 1.0
# This is the rate of the collection that will be used for kmeans sampling
KMEANS_SAMPLE_RATE = 1.0

NUM_CHUNKS_PER_ITEM = INPUT_LENGTH // CHUNK_LENGTH
assert NUM_CHUNKS_PER_ITEM * CHUNK_LENGTH == INPUT_LENGTH

# For Search
USE_GPU = True
