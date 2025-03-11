import logging
from src.document_sources.web_pages import get_documents_from_web_page
from dotenv import load_dotenv
from src.document_sources.youtube import get_documents_from_youtube
from src.document_sources.wikipedia import *
from src.document_sources.local_file import get_documents_from_file_by_path
from src.shared.utils import *
from dotenv import load_dotenv
import os

# Preprocess
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

my_value = os.getenv('LLM_MODEL_CONFIG_groq_llama3_70b')

print(f'MY_VARIABLE: {my_value}')

result = get_documents_from_file_by_path("D:\KerjaPraktik\Backend\datatest.csv", "datatest.csv")
print(result)

print("===================================== WIKIPEDIA SCRAPING ======================================")
wiki_result = get_documents_from_Wikipedia('Diabetes', 'id')
print(wiki_result)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info("Hello World")

print(get_documents_from_youtube("https://www.youtube.com/watch?v=IpFX2vq8HKw"))