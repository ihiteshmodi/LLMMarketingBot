import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import subprocess
import re

BASE_URL = "https://python.langchain.com"
GUIDES_URL = f"{BASE_URL}/docs/how_to/"

# Step 1: Scrape all guide URLs
def get_guide_links():
    response = requests.get(GUIDES_URL)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        if a.text.strip().lower().startswith("how to"):
            full_url = urljoin(BASE_URL, a['href'])
            links.append((a.text.strip(), full_url))

    return links

# Util: Clean up badly formatted code blocks
def clean_code_block(code):
    code = re.sub(r"[ \t]*\n[ \t]*", "\n", code)
    code = re.sub(r"\n{2,}", "\n\n", code)
    code = re.sub(r"(?<=\w)\n(?=\w)", " ", code)  # avoid breaking words
    return code.strip()

def clean_text(text):
    # Remove ANSI escape sequences and weird unicode like \u200b
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = text.replace('\u200b', '')
    return text.strip()

def transform_record(record):
    return {
        "instruction": clean_text(record["question"]),
        "input": clean_text(record["heading"]),
        "output": clean_text(record["answer"])
    }

# Step 2: Scrape one guide and extract title, content sections, code
def scrape_guide(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.find("h1").text.strip() if soup.find("h1") else "Untitled"
    sections = []
    current_heading = None
    current_content = []
    current_code_blocks = []

    for tag in soup.find_all(["h2", "p", "pre"]):
        if tag.name == "h2":
            if current_heading:
                section_text = " ".join(current_content).strip()
                formatted_code_blocks = [clean_code_block(cb) for cb in current_code_blocks]
                code_blocks_text = "\n\n".join([f"```python\n{cb}\n```" for cb in formatted_code_blocks])

                # Generate prompt for LLM
                prompt = f"""You are an expert in LangChain. Given the following section from the documentation, generate a question and a detailed answer that would help someone learn LangChain. Include code examples if present.

Section Heading: {current_heading}
Section Text: {section_text}
Code Blocks:
{code_blocks_text}

Format your response as:
Question: <question>
Answer: <answer>
"""

                try:
                    result = subprocess.run(
                        ["ollama", "run", "gemma3:4b"],
                        input="prompt",
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    output = result.stdout.strip()
                    if "Question:" in output and "Answer:" in output:
                        q = output.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
                        a = output.split("Answer:", 1)[1].strip()
                    else:
                        q = f"What is covered in the section '{current_heading}'?"
                        a = section_text + ("\n\n" + code_blocks_text if code_blocks_text else "")
                except Exception:
                    q = f"What is covered in the section '{current_heading}'?"
                    a = section_text + ("\n\n" + code_blocks_text if code_blocks_text else "")

                sections.append({
                    "heading": current_heading,
                    "question": q,
                    "answer": a
                })

            current_heading = tag.text.strip()
            current_content = []
            current_code_blocks = []

        elif tag.name == "p":
            current_content.append(tag.text.strip())
        elif tag.name == "pre":
            current_code_blocks.append(tag.get_text("\n", strip=False))

    # Add the final section
    if current_heading:
        section_text = " ".join(current_content).strip()
        formatted_code_blocks = [clean_code_block(cb) for cb in current_code_blocks]
        code_blocks_text = "\n\n".join([f"```python\n{cb}\n```" for cb in formatted_code_blocks])

        prompt = f"""You are an expert in LangChain. Given the following section from the documentation, generate a question and a detailed answer that would help someone learn LangChain. Include code examples if present.

Section Heading: {current_heading}
Section Text: {section_text}
Code Blocks:
{code_blocks_text}

Format your response as:
Question: <question>
Answer: <answer>
"""

        try:
            result = subprocess.run(
                ["ollama", "run", "gemma3:4b", "--prompt", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout.strip()
            if "Question:" in output and "Answer:" in output:
                q = output.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
                a = output.split("Answer:", 1)[1].strip()
            else:
                q = f"What is covered in the section '{current_heading}'?"
                a = section_text + ("\n\n" + code_blocks_text if code_blocks_text else "")
        except Exception:
            q = f"What is covered in the section '{current_heading}'?"
            a = section_text + ("\n\n" + code_blocks_text if code_blocks_text else "")

        sections.append({
            "heading": current_heading,
            "question": q,
            "answer": a
        })

    return {
        "title": title,
        "url": url,
        "sections": sections
    }


# Example usage
all_links = get_guide_links()
temp_link = all_links[0][1]
data = scrape_guide(temp_link)

with open("checking_transformed_dataset.jsonl", "w", encoding="utf-8") as fout:
    for section in data["sections"]:
        fout.write(json.dumps(transform_record(section), ensure_ascii=False) + "\n")
