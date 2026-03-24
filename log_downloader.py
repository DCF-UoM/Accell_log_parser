import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "http://130.88.33.69/DAFNE/?mode=summary&reverse=0&reverse=1&npp=3000&Entry+Type=%5EShift+Log%24"   #elog list with the correct flags, 3000 entries per page (make sure that every entry is on one page)
output_folder = "downloads"

os.makedirs(output_folder, exist_ok=True)

# Fetch the webpage
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

# Find all links that look like downloads
links = soup.find_all("a")

skip_strings = ["vacA", "vacB"]  # lowercase for easy matching

for link in links:
    href = link.get("href")

    if not href:
        continue

    # Full URL
    file_url = urljoin(url, href)

    #skip any file that contains any of the skip strings
    if any(s in file_url for s in skip_strings):
        print(f"Skipping {file_url} (contains skip string)")
        continue


    # Filter only file links
    if any(file_url.lower().endswith(ext) for ext in [
        ".pdf"
    ]):
        filename = os.path.join(output_folder, file_url.split("/")[-1])

        print(f"Downloading {file_url} → {filename}")

        file_data = requests.get(file_url)
        with open(filename, "wb") as f:
            f.write(file_data.content)