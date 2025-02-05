import re
import logging
from src.graph_query import *
from src.shared.constants import *
from src.shared.utils import time_to_seconds

def process_records(records):
    """[ENG]: Processes a record to extract and organize node and relationship data.
    [IDN]: Memproses record untuk mengekstrak dan mengatur data node dan data relasi."""
    try:
        nodes = []
        relationships = []
        seen_nodes = set()
        seen_relationships = set()

        for record in records:
            for element in record["entities"]:
                start_node = element['startNode']
                end_node = element['endNode']
                relationship = element['relationship']

                if start_node['element_id'] not in seen_nodes:
                    if "labels" in start_node.keys():
                        labels = set(start_node["labels"])
                        labels.discard("__Entity__")
                        if not labels:
                            labels.add('*')
                        start_node["labels"] = list(labels)
                    nodes.append(start_node)
                    seen_nodes.add(start_node['element_id'])

                if end_node['element_id'] not in seen_nodes:
                    if "labels" in end_node.keys():
                        labels = set(end_node["labels"])
                        labels.discard("__Entity__")
                        if not labels:
                            labels.add('*')
                        end_node["labels"] = list(labels)
                    nodes.append(end_node)
                    seen_nodes.add(end_node['element_id'])
                
                if relationship['element_id'] not in seen_relationships:
                    relationships.append({
                        "element_id": relationship['element_id'],
                        "type": relationship['type'],
                        "start_node_element_id": start_node['element_id'],
                        "end_node_element_id": end_node['element_id']
                    })
                    seen_relationships.add(relationship['element_id'])

        output = {
            "nodes": nodes,
            "relationships": relationships
        }

        return output

    except Exception as e:
        err = f"chunkid_entities module: Failed to extract the nodes and relationships from records with error: {str(e)}"
        logging.error(err)

def process_chunk_data(chunk_data):
    """[ENG]: Processes a record to extract chunk_text
    [IDN]: Memproses record untuk mengekstrak chunk_text"""
    try:
        required_doc_properties = ["fileSource", "fileType", "url"]
        chunk_properties = []

        for record in chunk_data:
            doc_properties = {prop: record["doc"].get(prop, None) for prop in required_doc_properties}
            for chunk in record["chunks"]:
                chunk.update(doc_properties)
                if chunk["fileSource"] == "youtube":
                    chunk["start_time"] = min(time_to_seconds(chunk.get('start_time', 0)), time_to_seconds(chunk.get('end_time', 0)))
                    chunk["end_time"] = time_to_seconds(chunk.get('end_time', 0))
                chunk_properties.append(chunk)
        
        return chunk_properties
    
    except Exception as e:
        err = f"chunkid_entities module: Failed extracting the Chunk text from records with error {e}"
        logging.error(err)

def remove_duplicate_nodes(nodes,property="element_id"):
    """[ENG]: Removes duplicate nodes based on a property.
    [IDN]: Menghapus node duplikat berdasarkan propertinya."""
    unique_nodes = []
    seen_element_ids = set()

    for node in nodes:
        element_id = node[property]
        if element_id not in seen_element_ids:
            if "labels" in node.keys():
                labels = set(node["labels"])
                labels.discard("__Entity__")
                if not labels:
                    labels.add('*')
                node["labels"] = list(labels)
            unique_nodes.append(node)
            seen_element_ids.add(element_id)

    return unique_nodes

def process_chunkids(driver, chunk_ids, entities):
    """[ENG] Processes chunk IDs to retrieve chunk data.
    [IDN] Memproses ID chunk untuk mengambil data chunk."""
    try:
        logging.info(f"Starting graph query process for chunk ids: {chunk_ids}")
        records, summary, keys = driver.execute_query(CHUNK_QUERY, chunksIds=chunk_ids,entityIds=entities["entityids"], relationshipIds=entities["relationshipids"])
        result = process_records(records)
        result["nodes"].extend(records[0]["nodes"])
        result["nodes"] = remove_duplicate_nodes(result["nodes"])
        logging.info(f"Nodes and relationships are processed")

        result["chunk_data"] = process_chunk_data(records)
        logging.info(f"Query process completed successfully for chunk ids: {chunk_ids}")
        return result

    except Exception as e:
        err = f"chunkid_entities module: Failed to process chunk ids {chunk_ids} with error: {str(e)}"
        logging.error(err)
        raise Exception(err)

def process_entityids(driver, entity_ids):
    """[ENG]: Processes entity IDs to retrieve local community data.
    [IDN]: Memproses ID entitas untuk mengambil data local community."""
    try:
        logging.info(f"Starting graph query process for entity ids: {entity_ids}")
        query_body = LOCAL_COMMUNITY_SEARCH_QUERY.format(
            topChunks = LOCAL_COMMUNITY_TOP_CHUNKS,
            topCommunities = LOCAL_COMMUNITY_TOP_COMMUNITIES,
            topOutsideRels = LOCAL_COMMUNITY_TOP_OUTSIDE_RELS
        )

        query = LOCAL_COMMUNITY_DETAILS_QUERY_PREFIX + query_body + LOCAL_COMMUNITY_DETAILS_QUERY_SUFFIX
        records, summary, keys = driver.execute_query(query, entityIds=entity_ids)
        result = process_records(records)

        if records:
            result["nodes"].extend(records[0]["nodes"])
            result["nodes"] = remove_duplicate_nodes(result["nodes"])

            logging.info(f"Nodes and relationships are processed")

            result["chunk_data"] = records[0]["chunks"]
            result["community_data"] = records[0]["communities"]
        else:
            result["chunk_data"] = list()
            result["community_data"] = list()
        logging.info(f"Query process completed successfully for chunk ids: {entity_ids}")
        return result
    except Exception as e:
        err = f"chunkid_entities module: Error processing entity ids {entity_ids} with error {e}"
        logging.error(err)
        raise Exception(err)
    
def process_communityids(driver, community_ids):
    """[ENG]: Processes community IDs to retrieve community data.
    [IDN]: Memproses ID community untuk mengambil data community."""
    try: 
        logging.info(f"Starting graph query process for community ids: {community_ids}")
        query = GLOBAL_COMMUNITY_DETAILS_QUERY
        records, summary, keys = driver.execute_query(query, communityIds=community_ids)

        result = {"nodes": [], "relationships": [], "chunk_data": []}
        result["community_data"] = records[0]["communities"] if records else []

        logging.info(f"Query process completed successfully for community ids: {community_ids}")
        return result
    except Exception as e:
        err = f"chunkid_entities module: Error processing community ids {community_ids} with error {e}"
        logging.error(err)
        raise Exception(err)
    
def get_entities_from_chunkids(uri, username, password, database , nodedetails, entities, mode):   
    try:

        driver = get_graphDB_driver(uri, username, password, database)
        default_response = {"nodes": list(),"relationships": list(),"chunk_data": list(),"community_data": list(),}

        nodedetails = json.loads(nodedetails)
        entities = json.loads(entities)

        if mode == CHAT_GLOBAL_VECTOR_FULLTEXT_MODE:

            if "communitydetails" in nodedetails and nodedetails["communitydetails"]:
                community_ids = [item["id"] for item in nodedetails["communitydetails"]]
                logging.info(f"chunkid_entities module: Starting for community ids: {community_ids}")
                return process_communityids(driver, community_ids)
            else:
                logging.info("chunkid_entities module: No community ids are passed")
                return default_response
            
        elif mode == CHAT_ENTITY_VECTOR_MODE:

            if "entitydetails" in nodedetails and nodedetails["entitydetails"]:
                entity_ids = [item["id"] for item in nodedetails["entitydetails"]]
                logging.info(f"chunkid_entities module: Starting for entity ids: {entity_ids}")
                result = process_entityids(driver, entity_ids)
                if "chunk_data" in result.keys():
                    for chunk in result["chunk_data"]:
                        chunk["text"] = re.sub(r'\s+', ' ', chunk["text"])
                return result
            else:
                logging.info("chunkid_entities module: No entity ids are passed")
                return default_response  
            
        else:

            if "chunkdetails" in nodedetails and nodedetails["chunkdetails"]:
                chunk_ids = [item["id"] for item in nodedetails["chunkdetails"]]
                logging.info(f"chunkid_entities module: Starting for chunk ids: {chunk_ids}")
                result = process_chunkids(driver, chunk_ids, entities)
                if "chunk_data" in result.keys():
                    for chunk in result["chunk_data"]:
                        chunk["text"] = re.sub(r'\s+', ' ', chunk["text"])
                return result
            else:
                logging.info("chunkid_entities module: No chunk ids are passed")
                return default_response

    except Exception as e:
        err = f"chunkid_entities module: Error occurred in get_entities_from_chunkids with error: {str(e)}"
        logging.error(err)
        raise Exception(f"chunkid_entities module: An error occurred in get_entities_from_chunkids.") from e 

