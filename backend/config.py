import os


DEFAULT_TARGET_MODELS = [
    "google/gemma-3n-e4b-it:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]


MODELS = {
    "intent_extractor": os.getenv("MODEL_INTENT", "google/gemma-3n-e4b-it:free"),
    "rewriter_a": os.getenv("MODEL_REWRITER_A", "google/gemma-3n-e4b-it:free"),
    "rewriter_b": os.getenv("MODEL_REWRITER_B", "openai/gpt-oss-20b:free"),
    "rewriter_c": os.getenv("MODEL_REWRITER_C", "meta-llama/llama-3.3-70b-instruct:free"),
    "reviewer_a": os.getenv("MODEL_REVIEWER_A", "google/gemma-3n-e4b-it:free"),
    "reviewer_b": os.getenv("MODEL_REVIEWER_B", "openai/gpt-oss-20b:free"),
    "reviewer_c": os.getenv("MODEL_REVIEWER_C", "meta-llama/llama-3.3-70b-instruct:free"),
    "chairman": os.getenv("MODEL_CHAIRMAN", "openai/gpt-oss-20b:free"),
}


TARGET_MODELS = [
    os.getenv("TARGET_MODEL_PRIMARY", DEFAULT_TARGET_MODELS[0]),
    os.getenv("TARGET_MODEL_SECONDARY", DEFAULT_TARGET_MODELS[1]),
    os.getenv("TARGET_MODEL_TERTIARY", DEFAULT_TARGET_MODELS[2]),
    os.getenv("TARGET_MODEL_ROUTER", DEFAULT_TARGET_MODELS[3]),
]
