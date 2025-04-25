import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
MODEL_NAME = "gpt-4o-mini"
client = OpenAI(api_key=API_KEY)

def lang_to_eng(text: str) -> str:
    """
    Automatically detect the input language of `text` and translate it into English.

    Args:
        text: A string in any language.

    Returns:
        The translated text in fluent English.
    """
    messages = [
        {"role": "system", "content": (
            "You are a translation assistant. Automatically detect the language of the user’s text, "
            "then translate it into clear, idiomatic English."
        )},
        {"role": "user", "content": text}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()


def eng_to_lang(text: str) -> str:
    """
    Translate input text from English into the project's source language.

    Args:
        text: A string in English.

    Returns:
        The translated text in the project's source language.
    """
    messages = [
        {"role": "system", "content": (
            "You are a translation assistant. Translate the user input from English into the project's source language, preserving nuance and context."
        )},
        {"role": "user", "content": text}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()
