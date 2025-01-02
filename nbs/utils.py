from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai  import AzureOpenAIEmbedding

from helper import get_azure_openai_keys
from llama_index.core import Settings



def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine


def get_router_query_engine_from_azure(file_path: str):


    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION = get_azure_openai_keys()

    llm = AzureOpenAI(
        engine="gpt-4o",
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    # You need to deploy your own embedding model as well as your own chat completion model
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    return get_router_query_engine(file_path, llm=llm, embed_model=embed_model)