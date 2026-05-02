import os


DEFAULT_TARGET_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openrouter/free",
]


MODELS = {
    "intent_extractor": os.getenv("MODEL_INTENT", "google/gemma-4-31b-it:free"),
    "rewriter_a": os.getenv("MODEL_REWRITER_A", "nvidia/nemotron-3-super-120b-a12b:free"),
    "rewriter_b": os.getenv("MODEL_REWRITER_B", "google/gemma-4-31b-it:free"),
    "rewriter_c": os.getenv("MODEL_REWRITER_C", "qwen/qwen3-next-80b-a3b-instruct:free"),
    "reviewer_a": os.getenv("MODEL_REVIEWER_A", "nvidia/nemotron-3-super-120b-a12b:free"),
    "reviewer_b": os.getenv("MODEL_REVIEWER_B", "google/gemma-4-31b-it:free"),
    "reviewer_c": os.getenv("MODEL_REVIEWER_C", "qwen/qwen3-next-80b-a3b-instruct:free"),
    "chairman": os.getenv("MODEL_CHAIRMAN", "nvidia/nemotron-3-super-120b-a12b:free"),
}


TARGET_MODELS = [
    os.getenv("TARGET_MODEL_PRIMARY", DEFAULT_TARGET_MODELS[0]),
    os.getenv("TARGET_MODEL_SECONDARY", DEFAULT_TARGET_MODELS[1]),
    os.getenv("TARGET_MODEL_TERTIARY", DEFAULT_TARGET_MODELS[2]),
    os.getenv("TARGET_MODEL_ROUTER", DEFAULT_TARGET_MODELS[3]),
]
