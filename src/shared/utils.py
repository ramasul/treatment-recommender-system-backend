import os
import re
import logging
import hashlib
from typing import List
from pathlib import Path
from urllib.parse import urlparse
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs.graph_document import GraphDocument
from src.document_sources.youtube import create_youtube_url

#Fungsi yang digunakan secara umum
def formatted_time(current_time):
  formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
  return str(formatted_time)

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

#Fungsi yang berkaitan dengan URL
def last_url_segment(url):
  parsed_url = urlparse(url)
  path = parsed_url.path.strip("/")  # Remove leading and trailing slashes
  last_url_segment = path.split("/")[-1] if path else parsed_url.netloc.split(".")[0]
  return last_url_segment

def check_url_source(source_type, yt_url:str=None, wiki_query:str=None):
    language=''
    try:
      logging.info(f"incoming URL: {yt_url}")
      if source_type == 'youtube':
        if re.match(r'(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?',yt_url.strip()):
          youtube_url = create_youtube_url(yt_url.strip())
          logging.info(youtube_url)
          return youtube_url, language
        else:
          raise Exception('Incoming URL is not youtube URL')
      
      elif  source_type == 'Wikipedia':
        wiki_query_id=''
        #pattern = r"https?:\/\/([a-zA-Z0-9\.\,\_\-\/]+)\.wikipedia\.([a-zA-Z]{2,3})\/wiki\/([a-zA-Z0-9\.\,\_\-\/]+)"
        wikipedia_url_regex = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
        wiki_id_pattern = r'^[a-zA-Z0-9 _\-\.\,\:\(\)\[\]\{\}\/]*$'
        
        match = re.search(wikipedia_url_regex, wiki_query.strip())
        if match:
                language = match.group(2)
                wiki_query_id = match.group(3)
          # else : 
          #       languages.append("en")
          #       wiki_query_ids.append(wiki_url.strip())
        else:
            raise Exception(f'Not a valid wikipedia url: {wiki_query} ')

        logging.info(f"wikipedia query id = {wiki_query_id}")     
        return wiki_query_id, language     
    except Exception as e:
      logging.error(f"Error in recognize URL: {e}")
      raise Exception(e)

#Fungsi yang berkaitan dengan database
def create_graph_database_connection(uri, userName, password, database):
  enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
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
  """[ENG]: Create a list of chunks and graph documents.
  [IDN]: Membuat daftar chunk dan dokumen graf."""
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
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logging.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
    else:
        err = f"Embedding model {embedding_model_name} is not supported"
        logging.error(err)
        raise Exception(err)
    return embeddings, dimension

def handle_backticks_nodes_relationship_id_type(graph_document_list:List[GraphDocument]):
    """[ENG]: Cleans node and relationship identifiers by removing backticks and ensuring 
    that only valid nodes and relationships with non-empty identifiers and types are retained.
    
    [IDN]: Membersihkan identifier node dan relationship dengan menghapus backtick serta memastikan 
    hanya node dan relationship yang valid dan tidak kosong dengan identifier dan tipenya yang dipertahankan.
    
    Args:
        graph_document_list (List[GraphDocument]): A list of graph documents containing nodes 
        and relationships.

    Returns:
        List[GraphDocument]: The cleaned list of graph documents with updated nodes and relationships."""
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

#Extras
def create_gcs_bucket_folder_name_hashed(uri, file_name):
  folder_name = uri + file_name
  folder_name_sha1 = hashlib.sha1(folder_name.encode())
  folder_name_sha1_hashed = folder_name_sha1.hexdigest()
  return folder_name_sha1_hashed
