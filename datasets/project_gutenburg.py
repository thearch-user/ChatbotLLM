# datasets/books/download.py
import requests
from bs4 import BeautifulSoup
import os

BASE = "https://www.gutenberg.org"
os.makedirs("datasets/books", exist_ok=True)

html = requests.get(f"{BASE}/browse/scores/top").text
soup = BeautifulSoup(html, "html.parser")

links = []
for a in soup.select("a[href^='/ebooks/']"):
    links.append(a["href"])

links = list(set(links))[:1000]  # 🔥 LIMIT SIZE

for link in links:
    book_id = link.split("/")[-1]
    txt_url = f"{BASE}/files/{book_id}/{book_id}-0.txt"
    r = requests.get(txt_url)
    if r.status_code == 200:
        with open(f"datasets/books/{book_id}.txt", "w", encoding="utf-8") as f:
            f.write(r.text)
