import os
import re
import logging
from typing import List
from pathlib import Path
from urllib.parse import urlparse
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs.graph_document import GraphDocument

#Fungsi yang digunakan secara umum
def formatted_time(current_time):
  formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
  return str(formatted_time)

#Fungsi yang berkaitan dengan URL
def last_url_segment(url):
  parsed_url = urlparse(url)
  path = parsed_url.path.strip("/")  # Remove leading and trailing slashes
  last_url_segment = path.split("/")[-1] if path else parsed_url.netloc.split(".")[0]
  return last_url_segment

# def check_url_source(source_type, yt_url:str=None, wiki_query:str=None):
#     language=''
#     try:
#       logging.info(f"incoming URL: {yt_url}")
#       if  source_type == 'Wikipedia':
#         wiki_query_id=''
#         wikipedia_url_regex = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
#         wiki_id_pattern = r'^[a-zA-Z0-9 _\-\.\,\:\(\)\[\]\{\}\/]*$'
        
#         match = re.search(wikipedia_url_regex, wiki_query.strip())
#         if match:
#                 language = match.group(2)
#                 wiki_query_id = match.group(3)

#         else:
#             raise Exception(f'Not a valid wikipedia url: {wiki_query} ')

#         logging.info(f"wikipedia query id = {wiki_query_id}")     
#         return wiki_query_id, language     
#     except Exception as e:
#       logging.error(f"Error in recognize URL: {e}")
#       raise Exception(e)

#Fungsi yang berkaitan dengan database
def create_graph_database_connection(uri, userName, password, database):
  enable_user_agent = os.getenv("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
  if enable_user_agent:
    graph = Neo4jGraph(url=uri, database=database, username=userName, password=password, refresh_schema=False, sanitize=True,driver_config={'user_agent':os.getenv('NEO4J_USER_AGENT')})  
  else:
    graph = Neo4jGraph(url=uri, database=database, username=userName, password=password, refresh_schema=False, sanitize=True)    
  return graph

def save_graphDocuments_in_neo4j(graph:Neo4jGraph, graph_document_list:List[GraphDocument]):
  graph.add_graph_documents(graph_document_list, baseEntityLabel=True)
  # graph.add_graph_documents(graph_document_list)

def close_db_connection(graph, api_name):
  if not graph._driver._closed:
      logging.info(f"closing connection for {api_name} api")
      graph._driver.close()   


def get_chunk_and_graphDocument(graph_document_list, chunkId_chunkDoc_list):
  logging.info("Creating list of chunks and graph documents in get_chunk_and_graphDocument func")
  lst_chunk_chunkId_document=[]
  for graph_document in graph_document_list:            
          for chunk_id in graph_document.source.metadata['combined_chunk_ids'] :
            lst_chunk_chunkId_document.append({'graph_doc':graph_document,'chunk_id':chunk_id})
                  
  return lst_chunk_chunkId_document  

def load_embedding_model(embedding_model_name: str):
    if embedding_model_name == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"#, cache_folder="/embedding_model"
        )
        dimension = 384
        logging.info(f"Embedding: Using Langchain HuggingFaceEmbeddings , Dimension:{dimension}")
    else:
        err = f"Embedding model {embedding_model_name} is not supported"
        logging.error(err)
        raise Exception(err)
    return embeddings, dimension

def handle_backticks_nodes_relationship_id_type(graph_document_list:List[GraphDocument]):
    for graph_document in graph_document_list:
      # Clean node id and types 
      cleaned_nodes = []
      for node in graph_document.nodes:
         if node.type.strip() and node.id.strip():
            node.type = node.type.replace('`', '')
            cleaned_nodes.append(node)

      # Clean relationship id types and source/target node id and types
      cleaned_relationships = []
      for rel in graph_document.relationships:
         if rel.type.strip() and rel.source.id.strip() and rel.source.type.strip() and rel.target.id.strip() and rel.target.type.strip():
            rel.type = rel.type.replace('`', '')
            rel.source.type = rel.source.type.replace('`', '')
            rel.target.type = rel.target.type.replace('`', '')
            cleaned_relationships.append(rel)
      graph_document.relationships = cleaned_relationships
      graph_document.nodes = cleaned_nodes
    return graph_document_list

def delete_uploaded_local_file(merged_file_path, file_name):
  file_path = Path(merged_file_path)
  if file_path.exists():
    file_path.unlink()
    logging.info(f'file {file_name} deleted successfully')


