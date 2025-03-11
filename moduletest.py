import json
import os
import shutil
import logging
import asyncio
import pandas as pd
from datetime import datetime as dt
from dotenv import load_dotenv
from api import *
from src.main import *
from src.QA_integration import QA_RAG
from langserve import add_routes
from graphdatascience import GraphDataScience
from src.entities.source_node import sourceNode

# Load environment variables if needed
load_dotenv()
# Constants
URI = os.environ.get('NEO4J_URI')
USERNAME = os.environ.get('NEO4J_USER')
PASSWORD = os.environ.get('NEO4J_PASSWORD')
DATABASE = os.environ.get('NEO4J_DATABASE')
BASE_DIR = os.path.dirname(__file__)
CHUNK_DIR = os.path.join(os.path.dirname(__file__), "chunks")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "merged_files")

# Initialize database connection
graph = create_graph_database_connection(URI, USERNAME, PASSWORD, DATABASE)

def test_graph_from_wikipedia(model_name):
    try:
       """Test graph creation from a Wikipedia page."""
       wiki_query = 'https://en.wikipedia.org/wiki/Ram_Mandir'
       source_type = 'Wikipedia'
       file_name = "Ram_Mandir"
       create_source_node_graph_url_wikipedia(graph, model_name, wiki_query, source_type)

       wiki_result = asyncio.run(extract_graph_from_file_Wikipedia(URI, USERNAME, PASSWORD, DATABASE, model_name, file_name, 'en',file_name, '', '',None, None))
       logging.info("Wikipedia test done")
       print(wiki_result)
       try:
           assert wiki_result['status'] == 'Completed'
           assert wiki_result['nodeCount'] > 0
           assert wiki_result['relationshipCount'] > 0
           print("Success")
       except AssertionError as e:
           print("Fail: ", e)
  
       return wiki_result
    except Exception as ex:
        print(ex)


# test_graph_from_wikipedia('diffbot')

def create_source_node_local(graph, model, file_name):
   """Creates a source node for a local file."""
   source_node = sourceNode()
   source_node.file_name = file_name
   source_node.file_type = 'pdf'
   source_node.file_size = '1087'
   source_node.file_source = 'local file'
   source_node.model = model
   source_node.created_at = dt.now()
   graphDB_data_Access = graphDBdataAccess(graph)
   graphDB_data_Access.create_source_node(source_node)
   return source_node

def test_graph_from_file_local(model_name):
   """Tests graph creation from a local file."""
   try:
       file_name = 'Clean_Daftar Penyakit dan Gejala.pdf'
       merged_file_path = os.path.join(MERGED_DIR, file_name)
       shutil.copyfile('./data/Clean/Clean_Daftar Penyakit dan Gejala.pdf', merged_file_path)
       graph = create_graph_database_connection(URI, USERNAME, PASSWORD, DATABASE)
       create_source_node_local(graph, model_name, file_name)
       result = asyncio.run(
           extract_graph_from_file_local_file(
               URI, USERNAME, PASSWORD, DATABASE, model_name, merged_file_path, file_name, '', '', None, ''
           )
       )
       logging.info(f"Local file test result: {result}")
       return result
   except Exception as e:
       logging.error(f"Error in test_graph_from_file_local: {e}")
       return {"status": "Failed", "error": str(e)}

# test_graph_from_file_local('diffbot')

def test_chatbot_qna(model_name, mode='vector'):
   """Test chatbot QnA functionality for different modes."""
   QA_n_RAG = QA_RAG(graph, model_name, 'Anda seorang dokter. Anda bertugas untuk diagnosis penyakit berdasarkan gejala pasien. Jelaskan penyakit tersebut dalam bahasa indonesia. Gejala yang dirasakan pasien: { sesak napas, nyeri dada, menggigil, batuk, detak jantung cepat, kelelahan, demam tinggi, tidak enak badan}', '[]', 1, mode)
   print(QA_n_RAG)
   print(len(QA_n_RAG['message']))


   try:
       assert len(QA_n_RAG['message']) > 20
       print("Success")
       print(QA_n_RAG)
       return QA_n_RAG
   
   except AssertionError as e:
       print("Failed ", e)
       return QA_n_RAG

# test_chatbot_qna('groq_llama3_70b', 'graph_vector_fulltext')
#print(asyncio.run(backend_connection_configuration()))