import os
import re
import logging
from typing import List
from urllib.parse import urlparse
from langchain_neo4j import Neo4jGraph
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

def check_url_source(source_type, yt_url:str=None, wiki_query:str=None):
    language=''
    try:
      logging.info(f"incoming URL: {yt_url}")
      if  source_type == 'Wikipedia':
        wiki_query_id=''
        wikipedia_url_regex = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
        wiki_id_pattern = r'^[a-zA-Z0-9 _\-\.\,\:\(\)\[\]\{\}\/]*$'
        
        match = re.search(wikipedia_url_regex, wiki_query.strip())
        if match:
                language = match.group(2)
                wiki_query_id = match.group(3)

        else:
            raise Exception(f'Not a valid wikipedia url: {wiki_query} ')

        logging.info(f"wikipedia query id = {wiki_query_id}")     
        return wiki_query_id, language     
    except Exception as e:
      logging.error(f"Error in recognize URL: {e}")
      raise Exception(e)

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
