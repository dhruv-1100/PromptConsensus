import os

MODELS = {
    "intent_extractor": os.getenv("MODEL_INTENT", "meta-llama/llama-3.2-3b-instruct:free"),
    "rewriter_a": os.getenv("MODEL_REWRITER_A", "meta-llama/llama-3.2-3b-instruct:free"),
    "rewriter_b": os.getenv("MODEL_REWRITER_B", "google/gemma-3-4b-it:free"),
    "rewriter_c": os.getenv("MODEL_REWRITER_C", "qwen/qwen3-4b:free"),
    "reviewer_a": os.getenv("MODEL_REVIEWER_A", "meta-llama/llama-3.2-3b-instruct:free"),
    "reviewer_b": os.getenv("MODEL_REVIEWER_B", "google/gemma-3-4b-it:free"),
    "reviewer_c": os.getenv("MODEL_REVIEWER_C", "qwen/qwen3-4b:free"),
    "chairman": os.getenv("MODEL_CHAIRMAN", "gemma-3-1b-it")
}
