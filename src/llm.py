import os
import logging
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.shared.constants import ADDITIONAL_INSTRUCTIONS

def get_llm(model: str):
    """[ENG]: Retrieve the specified language model based on the model name.
    [IDN]: Mendapatkan model LLM yang ditentukan berdasarkan nama modelnya.
    Model Option: groq, diffbot"""
    model = model.lower().strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    try:
        if "groq" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

        elif "diffbot" in model:
            model_name, api_key = env_value.split(",")
            llm = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"],
            )

    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name

def get_combined_chunks(chunkId_chunkDoc_list):
    """[ENG]: Combine the chunks based on the chunks to combine.
    [IDN]: Menggabungkan chunk berdasarkan jumlah chunk yang akan digabungkan dan pasangan Id-Dokumen."""
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
        Document(
            page_content=document["chunk_doc"].page_content,
            metadata={"chunk_id": [document["chunk_id"]]},
        )
        for document in chunkId_chunkDoc_list
    ]
    return combined_chunk_document_list

async def get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship, additional_instructions=None):
    futures = []
    graph_document_list = []
    if "diffbot_api_key" in dir(llm):
        llm_transformer = llm
    else:
        node_properties = ["description"]
        relationship_properties = ["description"]
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            ignore_tool_usage=True,
            additional_instructions=ADDITIONAL_INSTRUCTIONS+ (additional_instructions if additional_instructions else "")
        )

    if isinstance(llm,DiffbotGraphTransformer):
        graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
    else:
        graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    
    return graph_document_list

async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, additional_instructions=None):
    try:
        llm, model_name = get_llm(model)
        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)

        if allowedNodes is None or allowedNodes == "":
            allowedNodes = []
        else:
            allowedNodes = allowedNodes.split(',')
        
        if allowedRelationship is None or allowedRelationship == "":
            allowedRelationship = []
        else:
            allowedRelationship = allowedRelationship.split(',')
        
        graph_document_list = await get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship, additional_instructions)
        return graph_document_list
    
    except Exception as e:
        err = f"Error during extracting graph with llm: {e}"
        logging.error(err)
        raise Exception(err)