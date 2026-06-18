import os
import requests
import concurrent.futures
from threading import Lock
from config import *


os.makedirs(HTML_DIR, exist_ok=True)
# Lock for thread-safe writing to the error log
error_log_lock = Lock()

def log_error(original_line, error_message):
    with error_log_lock:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{original_line}\t{error_message}\n")

def download_paper(original_line):
    # Extract ID
    if "arXiv:" in original_line:
        paper_id = original_line.split("arXiv:")[-1].strip()
    else:
        paper_id = original_line.strip()
        
    url = BASE_URL.format(id=paper_id)
    output_path = os.path.join(HTML_DIR, f"{paper_id}.html")
    
    # If the file already exists, we skip it
    if os.path.exists(output_path):
        return True
        
    try:
        response = requests.get(
            url, 
            timeout=35, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
        )
        if response.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return True
        else:
            log_error(paper_id, f"HTTP {response.status_code}: {response.reason}")
            return False
    except Exception as e:
        log_error(paper_id, f"{type(e).__name__}: {e}")
        return False

def main(start_index=1, number_of_papers=None):
    if not os.path.exists(LIST_FILE):
        print(f"Error: {LIST_FILE} not found.")
        return

    with open(LIST_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if start_index < 1:
        print("Error: start_index must be 1 or greater.")
        return

    if number_of_papers is not None and number_of_papers < 1:
        print("Error: number_of_papers must be 1 or greater.")
        return

    start = start_index - 1
    end = None if number_of_papers is None else start + number_of_papers
    selected_lines = lines[start:end]

    if not selected_lines:
        print(f"No papers found starting at index {start_index}.")
        return

    # Initialize/clear error log for this run
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        pass

    print(f"Found {len(lines)} papers in list.")
    print(f"Processing {len(selected_lines)} papers starting at index {start_index}.")
    print("Starting download process... This may take a while depending on rate limits.")
    
    # Using ThreadPoolExecutor to download in parallel
    # Limiting to 5 workers to avoid overwhelming the server
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_paper, line): line for line in selected_lines}
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            if count % 20 == 0:
                print(f"Processed {count}/{len(selected_lines)} papers.")
                
    print(f"Download process finished. Any errors have been logged to {ERROR_LOG}.")

if __name__ == "__main__":
    main(1, 100)
