import requests
import csv
from datetime import datetime
import os

def query_crossref(query, rows=1000, offset=0):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows, "offset": offset}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Check for HTTP request errors
        data = response.json()

        # Extract metadata for each paper
        papers = []
        for item in data.get("message", {}).get("items", []):
            paper = {
                "doi":item.get("DOI", "No DOI"),
                "abstract": item.get("abstract", "No abstract available"),
                "label":query,
            }
            # Clean abstract: Remove HTML tags if any
            if paper["abstract"] != "No abstract available":
                paper["abstract"] = (
                    paper["abstract"]
                    .replace("<jats:p>", "")
                    .replace("</jats:p>", "")
                    .strip()
                )
                papers.append(paper)

        return papers
    except requests.exceptions.RequestException as e:
        print(f"Error querying CrossRef API: {e}")
        return []

def save_to_csv(filename, papers):
    fieldnames = ["doi", "abstract", "label"]
    
    try:
        # Check if file already exists to determine append or write mode
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file doesn't exist, write the header
            if not file_exists:
                writer.writeheader()
            
            # Append the rows of paper data
            writer.writerows(papers)
        
        print(f"Saved {len(papers)} papers to {filename}.")
    
    except IOError as e:
        print(f"Error saving to CSV: {e}")

def fetch_and_save_papers(query, total_papers, output_csv):
    papers_fetched = 0
    rows_per_query = 999
    offset = 0

    while papers_fetched < total_papers:
        # Fetch papers with current offset
        papers = query_crossref(query, rows=rows_per_query, offset=offset)
        if not papers:
            break
        
        # Save papers to CSV
        save_to_csv(output_csv, papers)
        
        # Update the number of papers fetched and the offset
        papers_fetched += len(papers)
        offset += rows_per_query
        
        print(f"Fetched {papers_fetched}/{total_papers} papers.")

if __name__ == "__main__":
    search_queries = {
        "psychology": 4000, 
        "sociology": 4000,
        "political science": 4000,
    }

    dir = os.path.join(os.getcwd(), str("data/"+datetime.today().strftime('%Y-%m-%d')))
    if not os.path.exists(dir):
        os.makedirs(dir)

    for search_query, total_papers in search_queries.items():
        # Save papers to CSV with pagination
        output_csv = str("data/"+datetime.today().strftime('%Y-%m-%d')) + "/" + search_query + ".csv"
        fetch_and_save_papers(search_query, total_papers, output_csv)
