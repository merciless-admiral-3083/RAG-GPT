# RAG Configuration Settings

# Distance threshold for relevance filtering
# Lower = stricter matching (fewer false positives)
# Higher = more lenient matching (more results)
BEST_DISTANCE_THRESHOLD = 1.5

# Number of chunks to retrieve initially
TOP_K_RETRIEVAL = 5

# Maximum chunks to use in final context
MAX_CHUNKS_RETURNED = 3

# Minimum word overlap between question and context
MIN_WORD_OVERLAP = 2

# Token limits for context
MAX_CONTEXT_TOKENS = 400

# Response quality settings
MIN_RESPONSE_LENGTH = 5  # minimum words
MAX_RESPONSE_SENTENCES = 4