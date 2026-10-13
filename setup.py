import os
from getpass import getpass



def get_pinecone_api_key():
    api_key = os.getenv("PINECONE_API_KEY").rstrip()
    if api_key is None:
        api_key = getpass("Enter your Pinecone API key: ")
    return api_key

def get_anthropic_api_key():
    api_key = os.getenv("ANTHROPIC_API_KEY").rstrip()
    if api_key is None:
        api_key = getpass("Enter your Anthropic API key: ")
    return api_key