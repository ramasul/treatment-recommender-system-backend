# Treatment Recommender System Backend
![Python](https://img.shields.io/badge/Python-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-green)

Welcome to the Treatment Recommender System Backend!

This is a project of building a recommender system using LLM integrated with knowledge graph. The knowledge graph is built using file or links, with fixed or non-fixed schema.
The Knowledge Graph used Leiden Algorithm for Community to enhance efficiency in querying.

The project is made for Industrial **Internship** Course in Department of Electrical Engineering and Information Technology (DEEIT)

## Features
- **Knowledge Graph Creation**: Transform unstructured data into structured knowledge graphs using LLMs.
- **Providing Schema**: Provide your own custom schema or use existing schema in settings to generate graph.
- CRUD operations for nodes and relationships
- Querying knowledge graph data
- Integration with Neo4j for graph storage

## Developer
This Backend Code is built by Rama Sulaiman Nurcahyo - 492727

## Requirements
Ensure you have the following installed:
- Python 3.8+

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ramasul/treatment-recommender-system-backend.git
   cd treatment-recommender-system-backend
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the API

1. Start the FastAPI server using Uvicorn:
   ```sh
   uvicorn score:app --reload
   ```

2. Open your browser and navigate to:
   - Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Redoc API docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Environment Variables
Create a `.env` file to store configurations like database credentials:
```ini
NEO4J_URI=""
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD=""
AURA_INSTANCEID=""
AURA_INSTANCENAME=""
LLM_MODEL_CONFIG_groq_llama3_70b="llama3-8b-8192,API-KEY" #Model Name, API Key
LLM_MODEL_CONFIG_diffbot="diffbot,API-KEY"
```

## API Endpoints
Example endpoints:
- `POST /url/scan` - Create source nodes for Neo4j database from source url
- `POST /extract` - Extract knowledge graph from file

