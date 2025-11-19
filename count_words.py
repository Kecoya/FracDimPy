import re

with open('paper.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove YAML front matter
content = re.sub(r'^---.*?---', '', content, flags=re.DOTALL)

# Remove markdown formatting but keep text
content = re.sub(r'[#*`\[\]()]', '', content)
content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove images
content = re.sub(r'\{.*?\}', '', content)  # Remove curly braces content

# Count words
words = len(content.split())
print(f'Estimated word count: {words}')