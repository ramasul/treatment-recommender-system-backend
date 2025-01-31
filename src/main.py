from document_sources.wikipedia import *
from document_sources.local_file import get_documents_from_file_by_path
from shared.utils import *
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables
my_value = os.getenv('GROQ_API_KEY')

print(f'MY_VARIABLE: {my_value}')

result = get_documents_from_file_by_path("D:\KerjaPraktik\Backend\datatest.csv", "datatest.csv")
print(result)
#get_documents_from_Wikipedia('https://en.wikipedia.org/wiki/Python_(programming_language)', 'en')