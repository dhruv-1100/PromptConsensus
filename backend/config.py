import os


DEFAULT_TARGET_MODELS = [
    "openai/gpt-5.4-nano",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v3.2",
    "nvidia/nemotron-3-super-120b-a12b",
]


MODELS = {
    "intent_extractor": os.getenv("MODEL_INTENT", "openai/gpt-5.4-nano"),
    "rewriter_a": os.getenv("MODEL_REWRITER_A", "google/gemini-2.5-flash"),
    "rewriter_b": os.getenv("MODEL_REWRITER_B", "openai/gpt-5.4-nano"),
    "rewriter_c": os.getenv("MODEL_REWRITER_C", "deepseek/deepseek-v3.2"),
    "reviewer_a": os.getenv("MODEL_REVIEWER_A", "google/gemini-2.5-flash"),
    "reviewer_b": os.getenv("MODEL_REVIEWER_B", "openai/gpt-5.4-nano"),
    "reviewer_c": os.getenv("MODEL_REVIEWER_C", "deepseek/deepseek-v3.2"),
    "chairman": os.getenv("MODEL_CHAIRMAN", "nvidia/nemotron-3-super-120b-a12b"),
}


TARGET_MODELS = [
    os.getenv("TARGET_MODEL_PRIMARY", DEFAULT_TARGET_MODELS[0]),
    os.getenv("TARGET_MODEL_SECONDARY", DEFAULT_TARGET_MODELS[1]),
    os.getenv("TARGET_MODEL_TERTIARY", DEFAULT_TARGET_MODELS[2]),
    os.getenv("TARGET_MODEL_ROUTER", DEFAULT_TARGET_MODELS[3]),
]
