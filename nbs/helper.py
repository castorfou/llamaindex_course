# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings


# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_azure_openai_keys():
    load_env()
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_api_version = os.getenv("AZURE_API_VERSION")
    return azure_api_key, azure_endpoint, azure_api_version


def get_azure_llm(engine="gpt-4o"):
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION = get_azure_openai_keys()
    llm = AzureOpenAI(
        engine=engine,
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )
    Settings.llm = llm
    return llm


def get_azure_embed_model(model="text-embedding-ada-002"):
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION = get_azure_openai_keys()
    embed_model = AzureOpenAIEmbedding(
        model=model,
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )
    Settings.embed_model = embed_model
    return embed_model
