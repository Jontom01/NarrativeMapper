import os

def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your shell or load it in your script using dotenv."
        )
    return key