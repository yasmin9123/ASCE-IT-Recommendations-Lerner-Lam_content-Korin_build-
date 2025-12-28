# this is a class definition 
from datetime import datetime
import hashlib
import json

class Memo:
    def __init__(self, title, author, content, date=None, attachments=None, tags=None):
        self.title = title
        self.author = author
        self.content = content
        self.date = date if date else datetime.now().strftime("%Y-%m-%d")
        self.attachments = attachments if attachments else []
        self.tags = tags if tags else []

        # Generate a SHA-256 hash of the content
        self.hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def update_content(self, new_content):
        self.content = new_content
        # Update hash whenever content changes
        self.hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def add_attachment(self, filename):
        self.attachments.append(filename)

    def add_tag(self, tag):
        self.tags.append(tag)

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "content": self.content,
            "attachments": self.attachments,
            "tags": self.tags,
            "hash": self.hash  # Include hash in the JSON export
        }

# Connect content from separate file
with open("content.txt", "r", encoding="utf-8") as f:
    memo_content = f.read()

# Create the Memo object
memo = Memo(
    title="Lerner-Lam AI-Readiness Memo to M.Lehman & M.Bomar",
    author="Eva Lerner-Lam",
    content=memo_content,
    date="2025-12-20"
)

# Add tags
memo.add_tag("AI Governance")
memo.add_tag("Civil Engineering")
memo.add_tag("Intelligence Tokens")
memo.add_tag("Soulbound Tokens")
memo.add_tag("Professional Ethics")
memo.add_tag("2025")
memo.add_tag("ASCE")
memo.add_tag("Eva Lerner-Lam")

# Attach images
memo.add_attachment("https://github.com/yasmin9123/ASCE-IT-Recommendations-Lerner-Lam_content-Korin_build-/raw/main/figure%201%20memo.png")
memo.add_attachment("https://github.com/yasmin9123/ASCE-IT-Recommendations-Lerner-Lam_content-Korin_build-/raw/main/figure%202%20memo.png")

# Export as JSON
with open("memo_2025_12_20.json", "w", encoding="utf-8") as f:
    json.dump(memo.to_dict(), f, indent=2)

print("Memo JSON created with hash:", memo.hash)
