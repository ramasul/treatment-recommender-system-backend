GROQ_MODELS = ["llama3-8b-8192"]
GRAPH_CHUNK_LIMIT = 50 

ADDITIONAL_INSTRUCTIONS = """Your goal is to identify and categorize entities while ensuring that specific data 
types such as dates, numbers, revenues, and other non-entity information are not extracted as separate nodes.
Instead, treat these as properties associated with the relevant entities."""

GRAPH_QUERY = """
MATCH docs = (d:Document) 
WHERE d.fileName IN $document_names
WITH docs, d 
ORDER BY d.createdAt DESC

// Fetch chunks for documents, currently with limit
CALL {{
  WITH d
  OPTIONAL MATCH chunks = (d)<-[:PART_OF|FIRST_CHUNK]-(c:Chunk)
  RETURN c, chunks LIMIT {graph_chunk_limit}
}}

WITH collect(distinct docs) AS docs, 
     collect(distinct chunks) AS chunks, 
     collect(distinct c) AS selectedChunks

// Select relationships between selected chunks
WITH *, 
     [c IN selectedChunks | 
       [p = (c)-[:NEXT_CHUNK|SIMILAR]-(other) 
       WHERE other IN selectedChunks | p]] AS chunkRels

// Fetch entities and relationships between entities
CALL {{
  WITH selectedChunks
  UNWIND selectedChunks AS c
  OPTIONAL MATCH entities = (c:Chunk)-[:HAS_ENTITY]->(e)
  OPTIONAL MATCH entityRels = (e)--(e2:!Chunk) 
  WHERE exists {{
    (e2)<-[:HAS_ENTITY]-(other) WHERE other IN selectedChunks
  }}
  RETURN entities, entityRels, collect(DISTINCT e) AS entity
}}

WITH docs, chunks, chunkRels, 
     collect(entities) AS entities, 
     collect(entityRels) AS entityRels, 
     entity

WITH *

CALL {{
  WITH entity
  UNWIND entity AS n
  OPTIONAL MATCH community = (n:__Entity__)-[:IN_COMMUNITY]->(p:__Community__)
  OPTIONAL MATCH parentcommunity = (p)-[:PARENT_COMMUNITY*]->(p2:__Community__) 
  RETURN collect(community) AS communities, 
         collect(parentcommunity) AS parentCommunities
}}

WITH apoc.coll.flatten(docs + chunks + chunkRels + entities + entityRels + communities + parentCommunities, true) AS paths

// Distinct nodes and relationships
CALL {{
  WITH paths 
  UNWIND paths AS path 
  UNWIND nodes(path) AS node 
  WITH distinct node 
  RETURN collect(node /* {{.*, labels:labels(node), elementId:elementId(node), embedding:null, text:null}} */) AS nodes 
}}

CALL {{
  WITH paths 
  UNWIND paths AS path 
  UNWIND relationships(path) AS rel 
  RETURN collect(distinct rel) AS rels 
}}  

RETURN nodes, rels

"""

CHUNK_QUERY = """
MATCH (chunk:Chunk)
WHERE chunk.id IN $chunksIds
MATCH (chunk)-[:PART_OF]->(d:Document)

WITH d, 
     collect(distinct chunk) AS chunks

// Collect relationships and nodes
WITH d, chunks, 
     collect {
         MATCH ()-[r]->() 
         WHERE elementId(r) IN $relationshipIds
         RETURN r
     } AS rels,
     collect {
         MATCH (e) 
         WHERE elementId(e) IN $entityIds
         RETURN e
     } AS nodes

WITH d, 
     chunks, 
     apoc.coll.toSet(apoc.coll.flatten(rels)) AS rels, 
     nodes

RETURN 
    d AS doc, 
    [chunk IN chunks | 
        chunk {.*, embedding: null, element_id: elementId(chunk)}
    ] AS chunks,
    [
        node IN nodes | 
        {
            element_id: elementId(node),
            labels: labels(node),
            properties: {
                id: node.id,
                description: node.description
            }
        }
    ] AS nodes,
    [
        r IN rels | 
        {
            startNode: {
                element_id: elementId(startNode(r)),
                labels: labels(startNode(r)),
                properties: {
                    id: startNode(r).id,
                    description: startNode(r).description
                }
            },
            endNode: {
                element_id: elementId(endNode(r)),
                labels: labels(endNode(r)),
                properties: {
                    id: endNode(r).id,
                    description: endNode(r).description
                }
            },
            relationship: {
                type: type(r),
                element_id: elementId(r)
            }
        }
    ] AS entities
"""

COUNT_CHUNKS_QUERY = """
MATCH (d:Document {fileName: $file_name})<-[:PART_OF]-(c:Chunk)
RETURN count(c) AS total_chunks
"""

CHUNK_TEXT_QUERY = """
MATCH (d:Document {fileName: $file_name})<-[:PART_OF]-(c:Chunk)
RETURN c.text AS chunk_text, c.position AS chunk_position, c.page_number AS page_number
ORDER BY c.position
SKIP $skip
LIMIT $limit
"""

### Vector graph search 
VECTOR_SEARCH_TOP_K = 5

VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, 
     collect(distinct {chunk: chunk, score: score}) AS chunks, 
     avg(score) AS avg_score

WITH d, avg_score, 
     [c IN chunks | c.chunk.text] AS texts, 
     [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails

WITH d, avg_score, chunkdetails, 
     apoc.text.join(texts, "\n----\n") AS text

RETURN text, 
       avg_score AS score, 
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" 
                             THEN d.fileName 
                             ELSE d.url 
                       END, 
                       d.fileName), 
        chunkdetails: chunkdetails} AS metadata
""" 


VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 40
VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH = 0.3
VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH = 0.9
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE = 20
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE = 40

VECTOR_GRAPH_SEARCH_QUERY_PREFIX = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)
// aggregate chunk-details
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score
// fetch entities
CALL { WITH chunks
UNWIND chunks as chunkScore
WITH chunkScore.chunk as chunk
"""

VECTOR_GRAPH_SEARCH_ENTITY_QUERY = """
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks 
    ORDER BY numChunks DESC 
    LIMIT {no_of_entites}

    WITH 
    CASE 
        WHEN e.embedding IS NULL OR ({embedding_match_min} <= vector.similarity.cosine($embedding, e.embedding) AND vector.similarity.cosine($embedding, e.embedding) <= {embedding_match_max}) THEN 
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document&!__Community__) 
                RETURN path LIMIT {entity_limit_minmax_case}
            }}
        WHEN e.embedding IS NOT NULL AND vector.similarity.cosine($embedding, e.embedding) >  {embedding_match_max} THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document&!__Community__) 
                RETURN path LIMIT {entity_limit_max_case} 
            }} 
        ELSE 
            collect {{ 
                MATCH path=(e) 
                RETURN path 
            }}
    END AS paths, e
"""

VECTOR_GRAPH_SEARCH_QUERY_SUFFIX = """
   WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
        collect(DISTINCT e) AS entities
   // De-duplicate nodes and relationships across chunks
   RETURN
       collect {
           UNWIND paths AS p
           UNWIND relationships(p) AS r
           RETURN DISTINCT r
       } AS rels,
       collect {
           UNWIND paths AS p
           UNWIND nodes(p) AS n
           RETURN DISTINCT n
       } AS nodes,
       entities
}
// Generate metadata and text components for chunks, nodes, and relationships
WITH d, avg_score,
    [c IN chunks | c.chunk.text] AS texts,
    [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,
    [n IN nodes | elementId(n)] AS entityIds,
    [r IN rels | elementId(r)] AS relIds,
    apoc.coll.sort([
        n IN nodes |
        coalesce(apoc.coll.removeAll(labels(n), ['__Entity__'])[0], "") + ":" +
        coalesce(
            n.id,
            n[head([k IN keys(n) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        ) +
        (CASE WHEN n.description IS NOT NULL THEN " (" + n.description + ")" ELSE "" END)
    ]) AS nodeTexts,
    apoc.coll.sort([
        r IN rels |
        coalesce(apoc.coll.removeAll(labels(startNode(r)), ['__Entity__'])[0], "") + ":" +
        coalesce(
            startNode(r).id,
            startNode(r)[head([k IN keys(startNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        ) + " " + type(r) + " " +
        coalesce(apoc.coll.removeAll(labels(endNode(r)), ['__Entity__'])[0], "") + ":" +
        coalesce(
            endNode(r).id,
            endNode(r)[head([k IN keys(endNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        )
    ]) AS relTexts,
    entities
// Combine texts into response text
WITH d, avg_score, chunkdetails, entityIds, relIds,
    "Text Content:\n" + apoc.text.join(texts, "\n----\n") +
    "\n----\nEntities:\n" + apoc.text.join(nodeTexts, "\n") +
    "\n----\nRelationships:\n" + apoc.text.join(relTexts, "\n") AS text,
    entities
RETURN
   text,
   avg_score AS score,
   {
       length: size(text),
       source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName),
       chunkdetails: chunkdetails,
       entities : {
           entityids: entityIds,
           relationshipids: relIds
       }
   } AS metadata
"""

VECTOR_GRAPH_SEARCH_QUERY = VECTOR_GRAPH_SEARCH_QUERY_PREFIX + VECTOR_GRAPH_SEARCH_ENTITY_QUERY.format(
    no_of_entites=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT,
    embedding_match_min=VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH,
    embedding_match_max=VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH,
    entity_limit_minmax_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE,
    entity_limit_max_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE
) + VECTOR_GRAPH_SEARCH_QUERY_SUFFIX

### Local community search
LOCAL_COMMUNITY_TOP_K = 10
LOCAL_COMMUNITY_TOP_CHUNKS = 3
LOCAL_COMMUNITY_TOP_COMMUNITIES = 3
LOCAL_COMMUNITY_TOP_OUTSIDE_RELS = 10

LOCAL_COMMUNITY_SEARCH_QUERY = """
WITH collect(node) AS nodes, 
     avg(score) AS score, 
     collect({{id: elementId(node), score: score}}) AS metadata

WITH score, nodes, metadata,

     collect {{
         UNWIND nodes AS n
         MATCH (n)<-[:HAS_ENTITY]->(c:Chunk)
         WITH c, count(distinct n) AS freq
         RETURN c
         ORDER BY freq DESC
         LIMIT {topChunks}
     }} AS chunks,

     collect {{
         UNWIND nodes AS n
         OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
         WITH c, c.community_rank AS rank, c.weight AS weight
         RETURN c
         ORDER BY rank, weight DESC
         LIMIT {topCommunities}
     }} AS communities,

     collect {{
         UNWIND nodes AS n
         UNWIND nodes AS m
         MATCH (n)-[r]->(m)
         RETURN DISTINCT r
         // TODO: need to add limit
     }} AS rels,

     collect {{
         UNWIND nodes AS n
         MATCH path = (n)-[r]-(m:__Entity__)
         WHERE NOT m IN nodes
         WITH m, collect(distinct r) AS rels, count(*) AS freq
         ORDER BY freq DESC 
         LIMIT {topOutsideRels}
         WITH collect(m) AS outsideNodes, apoc.coll.flatten(collect(rels)) AS rels
         RETURN {{ nodes: outsideNodes, rels: rels }}
     }} AS outside
"""

LOCAL_COMMUNITY_SEARCH_QUERY_SUFFIX = """
RETURN {
  chunks: [c IN chunks | c.text],
  communities: [c IN communities | c.summary],
  entities: [
    n IN nodes | 
    CASE 
      WHEN size(labels(n)) > 1 THEN 
        apoc.coll.removeAll(labels(n), ["__Entity__"])[0] + ":" + n.id + " " + coalesce(n.description, "")
      ELSE 
        n.id + " " + coalesce(n.description, "")
    END
  ],
  relationships: [
    r IN rels | 
    startNode(r).id + " " + type(r) + " " + endNode(r).id
  ],
  outside: {
    nodes: [
      n IN outside[0].nodes | 
      CASE 
        WHEN size(labels(n)) > 1 THEN 
          apoc.coll.removeAll(labels(n), ["__Entity__"])[0] + ":" + n.id + " " + coalesce(n.description, "")
        ELSE 
          n.id + " " + coalesce(n.description, "")
      END
    ],
    relationships: [
      r IN outside[0].rels | 
      CASE 
        WHEN size(labels(startNode(r))) > 1 THEN 
          apoc.coll.removeAll(labels(startNode(r)), ["__Entity__"])[0] + ":" + startNode(r).id + " "
        ELSE 
          startNode(r).id + " "
      END + 
      type(r) + " " +
      CASE 
        WHEN size(labels(endNode(r))) > 1 THEN 
          apoc.coll.removeAll(labels(endNode(r)), ["__Entity__"])[0] + ":" + endNode(r).id
        ELSE 
          endNode(r).id
      END
    ]
  }
} AS text,
score,
{entities: metadata} AS metadata
"""

LOCAL_COMMUNITY_DETAILS_QUERY_PREFIX = """
UNWIND $entityIds as id
MATCH (node) WHERE elementId(node) = id
WITH node, 1.0 as score
"""

LOCAL_COMMUNITY_DETAILS_QUERY_SUFFIX = """
WITH *
UNWIND chunks AS c
MATCH (c)-[:PART_OF]->(d:Document)
RETURN 
    [
        c {
            .*,
            embedding: null,
            fileName: d.fileName,
            fileSource: d.fileSource, 
            element_id: elementId(c)
        }
    ] AS chunks,
    [
        community IN communities WHERE community IS NOT NULL | 
        community {
            .*,
            embedding: null,
            element_id:elementId(community)
        }
    ] AS communities,
    [
        node IN nodes + outside[0].nodes | 
        {
            element_id: elementId(node),
            labels: labels(node),
            properties: {
                id: node.id,
                description: node.description
            }
        }
    ] AS nodes, 
    [
        r IN rels + outside[0].rels | 
        {
            startNode: {
                element_id: elementId(startNode(r)),
                labels: labels(startNode(r)),
                properties: {
                    id: startNode(r).id,
                    description: startNode(r).description
                }
            },
            endNode: {
                element_id: elementId(endNode(r)),
                labels: labels(endNode(r)),
                properties: {
                    id: endNode(r).id,
                    description: endNode(r).description
                }
            },
            relationship: {
                type: type(r),
                element_id: elementId(r)
            }
        }
    ] AS entities
"""

LOCAL_COMMUNITY_SEARCH_QUERY_FORMATTED = LOCAL_COMMUNITY_SEARCH_QUERY.format(
    topChunks=LOCAL_COMMUNITY_TOP_CHUNKS,
    topCommunities=LOCAL_COMMUNITY_TOP_COMMUNITIES,
    topOutsideRels=LOCAL_COMMUNITY_TOP_OUTSIDE_RELS) + LOCAL_COMMUNITY_SEARCH_QUERY_SUFFIX

### Global community search
GLOBAL_SEARCH_TOP_K = 10

GLOBAL_VECTOR_SEARCH_QUERY = """
WITH collect(distinct {community: node, score: score}) AS communities,
     avg(score) AS avg_score

WITH avg_score,
     [c IN communities | c.community.summary] AS texts,
     [c IN communities | {id: elementId(c.community), score: c.score}] AS communityDetails

WITH avg_score, communityDetails,
     apoc.text.join(texts, "\n----\n") AS text

RETURN text,
       avg_score AS score,
       {communitydetails: communityDetails} AS metadata
"""

GLOBAL_COMMUNITY_DETAILS_QUERY = """
MATCH (community:__Community__)
WHERE elementId(community) IN $communityids
WITH collect(distinct community) AS communities
RETURN [community IN communities | 
        community {.*, embedding: null, element_id: elementId(community)}] AS communities
"""

## CHAT MODES 

CHAT_VECTOR_MODE = "vector"
CHAT_FULLTEXT_MODE = "fulltext"
CHAT_ENTITY_VECTOR_MODE = "entity_vector"
CHAT_VECTOR_GRAPH_MODE = "graph_vector"
CHAT_VECTOR_GRAPH_FULLTEXT_MODE = "graph_vector_fulltext"
CHAT_GLOBAL_VECTOR_FULLTEXT_MODE = "global_vector"
CHAT_GRAPH_MODE = "graph"
CHAT_DEFAULT_MODE = "graph_vector_fulltext"

CHAT_MODE_CONFIG_MAP= {
        CHAT_VECTOR_MODE : {
            "retrieval_query": VECTOR_SEARCH_QUERY,
            "top_k": VECTOR_SEARCH_TOP_K,
            "index_name": "vector",
            "keyword_index": None,
            "document_filter": True,
            "node_label": "Chunk",
            "embedding_node_property":"embedding",
            "text_node_properties":["text"],

        },
        CHAT_FULLTEXT_MODE : {
            "retrieval_query": VECTOR_SEARCH_QUERY,  
            "top_k": VECTOR_SEARCH_TOP_K,
            "index_name": "vector",  
            "keyword_index": "keyword", 
            "document_filter": False,            
            "node_label": "Chunk",
            "embedding_node_property":"embedding",
            "text_node_properties":["text"],
        },
        CHAT_ENTITY_VECTOR_MODE : {
            "retrieval_query": LOCAL_COMMUNITY_SEARCH_QUERY_FORMATTED,
            "top_k": LOCAL_COMMUNITY_TOP_K,
            "index_name": "entity_vector",
            "keyword_index": None,
            "document_filter": False,            
            "node_label": "__Entity__",
            "embedding_node_property":"embedding",
            "text_node_properties":["id"],
        },
        CHAT_VECTOR_GRAPH_MODE : {
            "retrieval_query": VECTOR_GRAPH_SEARCH_QUERY,
            "top_k": VECTOR_SEARCH_TOP_K,
            "index_name": "vector",
            "keyword_index": None,
            "document_filter": True,            
            "node_label": "Chunk",
            "embedding_node_property":"embedding",
            "text_node_properties":["text"],
        },
        CHAT_VECTOR_GRAPH_FULLTEXT_MODE : {
            "retrieval_query": VECTOR_GRAPH_SEARCH_QUERY,
            "top_k": VECTOR_SEARCH_TOP_K,
            "index_name": "vector",
            "keyword_index": "keyword",
            "document_filter": False,            
            "node_label": "Chunk",
            "embedding_node_property":"embedding",
            "text_node_properties":["text"],
        },
        CHAT_GLOBAL_VECTOR_FULLTEXT_MODE : {
            "retrieval_query": GLOBAL_VECTOR_SEARCH_QUERY,
            "top_k": GLOBAL_SEARCH_TOP_K,
            "index_name": "community_vector",
            "keyword_index": "community_keyword",
            "document_filter": False,            
            "node_label": "__Community__",
            "embedding_node_property":"embedding",
            "text_node_properties":["summary"],
        },
    }

YOUTUBE_CHUNK_SIZE_SECONDS = 60

