"""
Shared RAG code-database loader/builder -- factored out so every pipeline
script in this directory that needs the persisted, dynamic-splitting
code-centric-chunk database (see PyMuPDF.py) can load it the same way,
instead of each re-parsing all 10 textbooks and re-embedding every chunk.
"""
import os
import sys

# This system's built-in sqlite3 (3.34.1) is below Chroma's minimum (3.35.0);
# swap in pysqlite3-binary's bundled modern build before chromadb (imported
# transitively by langchain.vectorstores.Chroma below) checks the version.
# See https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from PyMuPDF import get_pdf_blocks, block_to_text

# Only code-centric chunks (dynamic splitting: code block + its immediate
# neighboring paragraphs) ever get retrieved from -- see HybridRetrieval and
# the syntax checker in the pipeline scripts, both of which only ever use
# the code retriever. Persisted here so repeat runs load the existing
# embeddings from disk instead of re-parsing all 10 textbooks and
# re-embedding every chunk.
CODE_DB_PERSIST_DIR = "RAG_Database/code_blocks_all_textbooks"

# IEEE Std 1800-2017's own Chapter 16 ("Assertions") -- the ground-truth
# source for SVA operator semantics/precedence -- extracted out of the full
# 1315-page standard (RelatedWorks/IEEE Standard for SystemVerilog.pdf,
# pages 364-483) into its own PDF, so it can be added to the same textbook
# corpus without pulling in the ~1200 unrelated pages covering the rest of
# the SystemVerilog LRM (classes, coverage, DPI-C, ...). Kept as a
# script-local addition rather than editing the shared
# VerilogTextBooks/AllTextbooks.txt, since other pipeline variants build
# their own separate (unpersisted) databases from that same file.
EXTRA_TEXTBOOK_NAMES = ["IEEE1800-2017-Ch16-Assertions"]


def build_rag_system(pdf_txt_filename, openai_api_key):
    """
    1. If a persisted code-block database already exists on disk, load it
       directly -- no PDF parsing, no re-embedding.
    2. Otherwise, extract code blocks (each packaged with its immediately
       preceding/following paragraph -- the dynamic-splitting code-centric
       chunk) from every textbook in pdf_txt_filename plus
       EXTRA_TEXTBOOK_NAMES, embed them, and persist the result to
       CODE_DB_PERSIST_DIR for next time.
    """
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)  # or HuggingFaceEmbeddings(), etc.

    if os.path.isdir(CODE_DB_PERSIST_DIR) and os.listdir(CODE_DB_PERSIST_DIR):
        return Chroma(
            persist_directory=CODE_DB_PERSIST_DIR,
            embedding_function=embedding_fn,
            collection_name="code_blocks",
        )

    with open(f"VerilogTextBooks/{pdf_txt_filename}") as file:
        pdf_names = [line.strip() for line in file.readlines()]
    pdf_names += EXTRA_TEXTBOOK_NAMES

    # Loop over each PDF name provided in the list
    blocks = []
    for pdf_name in pdf_names:
        # Construct the file path for the current PDF
        pdf_path = f"VerilogTextBooks/{pdf_name}.pdf"
        # Get all blocks from PDF
        blocks += get_pdf_blocks(pdf_path)

    # Keep only code blocks, each packaged with its neighboring paragraphs
    code_docs = []
    for idx, block in enumerate(blocks):
        if block["type"] != "code":
            continue
        last_content = "" if idx == 0 else block_to_text(blocks[idx - 1])
        next_content = "" if idx == len(blocks) - 1 else block_to_text(blocks[idx + 1])
        content = f"{last_content}\n\n{block_to_text(block)}\n\n{next_content}"
        code_docs.append(Document(page_content=content, metadata={"type": "code", "block_index": idx}))

    return Chroma.from_documents(
        code_docs,
        embedding=embedding_fn,
        collection_name="code_blocks",
        persist_directory=CODE_DB_PERSIST_DIR,
    )
