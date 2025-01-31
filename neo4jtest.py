import os
import time
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

neo4j_configuration = [
    {
        'name': 'Neo4j Config 1',
        'NEO4J_URI': os.getenv('NEO4J_URI'),
        'NEO4J_USERNAME': os.getenv('NEO4J_USERNAME'),
        'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD')
    },
]

logging.info(neo4j_configuration)

# Connect to Neo4j
def create_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

# Delete all the nodes and relationships in the database
def clear_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

# Performance test function
def performance_test(driver, query, num_operations):
    with driver.session() as session:
        start_time = time.time()
        for i in range(num_operations):
            session.run(query, parameters={"id": i, "name": f"name_{i}"})
        end_time = time.time()
    return end_time - start_time

# Query to test
query = "CREATE (n:Node {id: $id, name: $name})"

# Number of operations to test
num_operations = 1000

# Main function for testing
def neo4jtest_main():
    results = []

    for config in neo4j_configuration:
        # Create driver
        driver = create_driver(config['NEO4J_URI'], config['NEO4J_USERNAME'], config['NEO4J_PASSWORD'])

        # Clear database before testing
        clear_database(driver)

        # Run performance test
        elapsed_time = performance_test(driver, query, num_operations)

        # Append results
        results.append((config['name'], elapsed_time))

        # Close driver
        driver.close()

        print(f"{config['name']} finished in {elapsed_time:.4f} seconds")

    print("\nPerfomance Test Results:")
    for name, time_taken in results:
        print(f"{name}: {time_taken:.4f} seconds")

if __name__ == "__main__":
    neo4jtest_main()