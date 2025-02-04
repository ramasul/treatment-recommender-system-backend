import os
import json
import logging
from neo4j import time
from neo4j import GraphDatabase
from src.shared.constants import GRAPH_CHUNK_LIMIT, GRAPH_QUERY, CHUNK_TEXT_QUERY, COUNT_CHUNKS_QUERY

def get_graphDB_driver(uri, username, password, database = "neo4j"):
    """
    [ENG]: Creates and returns a Neo4j database driver instance configured with the provided credentials.
    [IDN]: Membuat dan mendapatkan instance driver database Neo4j yang dikonfigurasi dengan kredensial yang diberikan.

    Returns:
    Neo4j.Driver: A driver object for interacting with the Neo4j database."""

    try:
        logging.info(f"Attempting to connect to the Neo4j database at {uri}")
        enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
        if enable_user_agent:
            driver = GraphDatabase.driver(uri, auth = (username, password), database = database, user_agent = os.environ.get('NEO4J_USER_AGENT'))
        else:
            driver = GraphDatabase.driver(uri, auth=(username, password),database=database)
        logging.info("Connected Successfully")
        return driver
    
    except Exception as e:
        err = f"graph_query module: Failed to connect to the database at {uri} with error: {str(e)}"
        logging.error(err, exc_info=True)

def execute_query(driver, query, document_names, doc_limit = None):
    """
    [ENG]: Executes a specified query using the Neo4j driver, with parameters based on the presence of a document name.
    [IDN]: Menjalankan query yang diinputkan menggunakan driver Neo4j, dengan parameter berdasarkan nama dokumen.

    Returns:
    tuple: Contains records, summary of the execution, and keys of the records."""
    
    try:
        if document_names:
            logging.info(f"Executing query for documents: {document_names}")
            records, summary, keys = driver.execute_query(query, document_names = document_names)
        else:
            logging.info(f"Executing query with a document limit of {doc_limit}")
            records, summary, keys = driver.execute_query(query, doc_limit = doc_limit)
        return records, summary, keys
    
    except Exception as e:
        err = f"graph_query module: Failed to execute the query with error: {str(e)}"
        logging.error(err, exc_info = True)

def process_node(node):
    """
    [ENG]: Processes a node from a Neo4j database, extracting its ID, labels, and properties,
    while omitting certain properties like 'embedding' and 'text'.
    [IDN]: Memproses node dari database Neo4j, mengekstrak ID, label, dan properti,

    Returns:
    dict: A dictionary with the node's element ID, labels, and other properties,
          with datetime objects formatted as ISO strings.
    """
    
    try:
        labels = set(node.labels)
        labels.discard("__Entity__")
        if not labels:
            labels.add('*')
        
        node_element = {
            "element_id": node.element_id,
            "labels": list(labels),
            "properties": {}
        }
        # logging.info(f"Processing node with element ID: {node.element_id}")

        for key in node:
            if key in ["embedding", "text", "summary"]:
                continue
            value = node.get(key)
            if isinstance(value, time.DateTime):
                node_element["properties"][key] = value.isoformat()
            else:
                node_element["properties"][key] = value

        return node_element
    except Exception as e:
        logging.error("graph_query module: An unexpected error occured while processing the node")

def extract_node_elements(records):
    """[ENG]: Extracts and processes unique nodes from a list of records, avoiding duplication by tracking seen element IDs.
    [IDN]: Mengekstrak dan memproses node yang unik dari daftar records, menghindari duplikasi dengan melacak ID elemen yang terlihat.

    Returns:
    list of dict: A list containing processed node dictionaries.
    """
    node_elements = []
    seen_element_ids = set()

    try:
        for record in records:
            nodes = record.get("nodes", [])
            if not nodes:
                continue

            for node in nodes:
                if node.element_id in seen_element_ids:
                    continue
                seen_element_ids.add(node.element_id)
                node_element = process_node(node)
                node_elements.append(node_element)
                # logging.info(f"Processed node with element ID: {node.element_id}")
        return node_elements
    
    except Exception as e:
        err = f"graph_query module: An unexpected error occured while extracting node elements: {str(e)}"
        logging.error(err, exc_info = True)
        raise Exception(err)
    
def extract_relationships(records):
    """[ENG]: Extracts and processes relationships from a list of records, ensuring that each relationship is processed only once by tracking seen element IDs.
    [IDN]: Mengekstrak dan memproses relasi dari daftar records, memastikan bahwa setiap relasi diproses sekali saja dengan melacak ID elemen yang terlihat.

    Returns:
    list of dict: A list containing dictionaries of processed relationships.
    """
    all_relationships = []
    seen_element_ids = set()

    try:
        for record in records:
            relationships = []
            relations = record.get("rels", [])
            if not relations:
                continue

            for relation in relations:
                if relation.element_id in seen_element_ids:
                    continue
                seen_element_ids.add(relation.element_id)

                try:
                    nodes = relation.nodes
                    if len(nodes) < 2:
                        logging.warning(f"Relationship with ID {relation.element_id} does not have two nodes.")
                        continue

                    relationship = {
                        "element_id": relation.element_id,
                        "type": relation.type,
                        "start_node_element_id": process_node(nodes[0])["element_id"],
                        "end_node_element_id": process_node(nodes[1])["element_id"],
                    }

                    relationships.append(relationship)
                
                except Exception as inner_e:
                    err = f"graph_query module: Failed to process relationship with ID {relation.element_id} with error {inner_e}"
                    logging.error(err, exc_info = True)
            all_relationships.extend(relationships)
        
        return all_relationships
    
    except Exception as e:
        err = f"graph_query module: An error occurred while extracting relationships from records with error: {str(e)}"
        logging.error(err, exc_info = True)

def get_completed_documents(driver):
    """[ENG]: Retrieves the names of all documents with the status 'Completed' from the database.
    [IDN]: Mengambil nama semua dokumen dengan status 'Completed'/'Selesai' dari database.
    """
    docs_query = "MATCH(node:Document {status:'Completed'}) RETURN node"
    
    try:
        logging.info("Executing query to retrieve completed documents")
        records, summary, keys = driver.execute_query(docs_query)
        logging.info(f"Query executed successfully, retrieved {len(records)} records.")
        documents = [record["node"]["fileName"] for record in records]
        logging.info("Document names extracted successfully.")
    
    except Exception as e:
        err = f"graph_query module: An error occured: {str(e)}"
        logging.error(err, exc_info = True)
    
    return documents

def get_graph_results(uri, username, password, database, document_names):
    """
    [ENG]: Retrieves graph data by executing a specified Cypher query using credentials and parameters provided.
    Processes the results to extract nodes and relationships and packages them in a structured output.
    [IDN]: Mengambil data graf dengan menjalankan query Cypher yang ditentukan menggunakan kredensial dan parameter yang diberikan.
    Memproses hasil untuk mengekstrak node dan relasi dan mengemasnya dalam output terstruktur.
    
    Args:
    uri (str): The URI for the Neo4j database.
    username (str): The username for authentication.
    password (str): The password for authentication.
    query_type (str): The type of query to be executed.
    document_name (str, optional): The name of the document to specifically query for, if any. Default is None.

    Returns:
    dict: Contains the session ID, user-defined messages with nodes and relationships, and the user module identifier.
    """
    try:
        logging.info(f"Starting graph query process")
        driver = get_graphDB_driver(uri, username, password, database)
        document_names = list(map(str, json.loads(document_names)))
        query = GRAPH_QUERY.format(graph_chunk_limit = GRAPH_CHUNK_LIMIT)
        records, summary, keys = execute_query(driver, query.strip(), document_names)
        document_nodes = extract_node_elements(records)
        document_relationships = extract_relationships(records)

        logging.info(f"No of nodes: {len(document_nodes)}")
        logging.info(f"No of relations: {len(document_relationships)}")
        result = {
            "nodes": document_nodes,
            "relationships": document_relationships
        }

        logging.info(f"Query process completed successfully")
        return result 

    except Exception as e:
        err = f"graph_query module: An error occured while executing the graph query: {str(e)}"
        logging.error(err, exc_info = True)
        raise Exception(err)

    finally:
        logging.info("Closing the database connection for graph_query api module")
        driver.close()

def get_chunktext_results(uri, username, password, database, document_name, page_no):
    """[ENG]: Retrieves chunk text, position, and page number from graph data with pagination.
    [IDN]: Mendapatkan teks, posisi, dan nomor halaman potongan dari data graf dengan penomoran halaman."""
    driver = None
    try:
        logging.info("Starting chunk text query process")
        offset = 10
        skip = (page_no - 1) * offset
        limit = offset
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
           total_chunks_result = session.run(COUNT_CHUNKS_QUERY, file_name=document_name)
           total_chunks = total_chunks_result.single()["total_chunks"]
           total_pages = (total_chunks + offset - 1) // offset  # Calculate total pages
           records = session.run(CHUNK_TEXT_QUERY, file_name=document_name, skip=skip, limit=limit)
           pageitems = [
               {
                   "text": record["chunk_text"],
                   "position": record["chunk_position"],
                   "pagenumber": record["page_number"]
               }
               for record in records
           ]
           logging.info(f"Query process completed with {len(pageitems)} chunks retrieved")
           return {
               "pageitems": pageitems,
               "total_pages": total_pages
           }
    except Exception as e:
       logging.error(f"An error occurred in get_chunktext_results. Error: {str(e)}")
       raise Exception("An error occurred in get_chunktext_results. Please check the logs for more details.") from e
    finally:
       if driver:
           driver.close()