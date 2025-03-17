import logging
from src.document_sources.web_pages import get_documents_from_web_page
from dotenv import load_dotenv
from src.document_sources.youtube import get_documents_from_youtube
from src.document_sources.wikipedia import *
from src.document_sources.local_file import get_documents_from_file_by_path
from src.shared.utils import *
from dotenv import load_dotenv
import os
from src.main import connection_check_and_get_vector_dimensions
from src.QA_integration import QA_RAG

# Preprocess
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')
uri = os.getenv('NEO4J_URI')
userName = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')
database = os.getenv('NEO4J_DATABASE')
graph = create_graph_database_connection(uri, userName, password, database)
print(connection_check_and_get_vector_dimensions(graph, "neo4j"))
model="openai_gpt_4o_mini"
question= "Anda seorang dokter. Anda bertugas untuk diagnosis penyakit berdasarkan gejala pasien. Jelaskan penyakit tersebut, perawatan, dan obatnya. Gejala yang dirasakan pasien: { Pusing, panas, demam, pilek}"
session_id = "123"
mode = "graph_vector_fulltext"
document_names = "[]"
print(QA_RAG(graph, model, question, document_names, session_id, mode, write_access=True))

from src.QA_integration import *
# initialize_neo4j_vector(graph, get_chat_mode_settings(mode))
# my_value = os.getenv('LLM_MODEL_CONFIG_groq_llama3_70b')

# print(f'MY_VARIABLE: {my_value}')

# result = get_documents_from_file_by_path("D:\KerjaPraktik\Backend\datatest.csv", "datatest.csv")
# print(result)

# print("===================================== WIKIPEDIA SCRAPING ======================================")
# wiki_result = get_documents_from_Wikipedia('Diabetes', 'id')
# print(wiki_result)

# load_dotenv()
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logging.info("Hello World")

# print(get_documents_from_youtube("https://www.youtube.com/watch?v=IpFX2vq8HKw"))