import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

pages = [
    "Artificial intelligence",
    "Neural network",
    "Machine learning",
    "Computer science",
    "Philosophy"
]

out = open("datasets/wiki.txt", "w", encoding="utf-8")

for title in pages:
    page = wiki.page(title)
    if page.exists():
        out.write(page.text + "\n")

out.close()
print("Wikipedia dataset saved.")
