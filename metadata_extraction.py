import json
import re
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# ----------------------------------------------------
# Load entire parsed Constitution text
# ----------------------------------------------------
with open("output/parsed.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

full_text = "\n".join([entry["text"] for entry in raw_docs])
lines = full_text.split("\n")


# ----------------------------------------------------
# STEP 1: Chunk WITHOUT metadata (keeps chunk count same)
# ----------------------------------------------------
big_doc = Document(text=full_text)

splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents([big_doc])

print(f"Chunks generated: {len(nodes)}")


# ----------------------------------------------------
# STEP 2: Detect PART + ARTICLE boundaries with line numbers
# ----------------------------------------------------
part_pattern = re.compile(r"^\s*PART\s+([IVXLC]+)\s*$", re.IGNORECASE)
article_pattern = re.compile(r"^\s*(\d+[A-Z]?)\.\s*(.*)$")

article_boundaries = []   # list of dicts with start_line and metadata
current_part = None
current_part_name = None
current_article = None
current_title = None

for idx, line in enumerate(lines):
    s = line.strip()

    # PART detection
    m_part = part_pattern.match(s)
    if m_part:
        current_part = m_part.group(1)

        # part name is next non-empty non-"ARTICLES" line
        for j in range(idx + 1, min(idx + 6, len(lines))):
            nxt = lines[j].strip()
            if nxt and nxt.upper() != "ARTICLES":
                current_part_name = nxt
                break

    # ARTICLE detection
    m_art = article_pattern.match(s)
    if m_art:
        current_article = m_art.group(1)
        current_title = m_art.group(2).strip() or None

        # register new article boundary
        article_boundaries.append({
            "start_line": idx,
            "part_number": current_part,
            "part_name": current_part_name,
            "article_number": current_article,
            "article_title": current_title,
            "source": "Constitution of India"
        })

# Add a sentinel end boundary
article_boundaries.append({"start_line": len(lines) + 1})


# ----------------------------------------------------
# STEP 3: For each chunk → determine line range → find correct article
# ----------------------------------------------------
def get_line_number_of_phrase(phrase):
    """Find approximate first occurrence line number of chunk."""
    for i, line in enumerate(lines):
        if phrase in line:
            return i
    return None  # fallback


final_nodes = []

for node in nodes:
    text = node.get_content()

    # get first non-empty line in chunk
    first_line = None
    for ln in text.split("\n"):
        if ln.strip():
            first_line = ln.strip()
            break

    # find approx line number in original text
    line_num = get_line_number_of_phrase(first_line)

    # fallback metadata
    chosen_meta = {
        "part_number": None,
        "part_name": None,
        "article_number": None,
        "article_title": None,
        "source": "Constitution of India"
    }

    if line_num is not None:
        # find the article that this line falls under
        for i in range(len(article_boundaries) - 1):
            a = article_boundaries[i]
            b = article_boundaries[i + 1]

            if a["start_line"] <= line_num < b["start_line"]:
                chosen_meta = {
                    "part_number": a["part_number"],
                    "part_name": a["part_name"],
                    "article_number": a["article_number"],
                    "article_title": a["article_title"],
                    "source": "Constitution of India"
                }
                break

    # attach metadata
    node.metadata = chosen_meta
    final_nodes.append(node)


# ----------------------------------------------------
# STEP 4: Save output
# ----------------------------------------------------
output = [
    {"text": n.get_content(), "metadata": n.metadata}
    for n in final_nodes
]

with open("output/structured_docs.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ Metadata applied safely and chunk count remains identical.")
