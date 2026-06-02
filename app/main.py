import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI # type: ignore

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    is_local = os.getenv("local") == "true"
    
    model = (
        "nvidia/nemotron-3-super-120b-a12b:free"
        if is_local
        else "anthropic/claude-haiku-4.5"
    )
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.p}],
        max_tokens = 1000,
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the following line to pass the first stage
    print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
