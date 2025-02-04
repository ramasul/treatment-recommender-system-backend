GROQ_MODELS = ["llama3-8b-8192"]
GRAPH_CHUNK_LIMIT = 50 

ADDITIONAL_INSTRUCTIONS = """Your goal is to identify and categorize entities while ensuring that specific data 
types such as dates, numbers, revenues, and other non-entity information are not extracted as separate nodes.
Instead, treat these as properties associated with the relevant entities."""