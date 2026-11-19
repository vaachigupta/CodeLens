"""
chain.py - CodeLens review pipeline built with LangChain (Unit III).

The whole review is ONE chain, written with LangChain's pipe (|) syntax:

    new code
       |
       v
    [Retriever]  -> finds the most similar functions in the indexed codebase
       |
       v
    [Prompt]     -> puts the references + new code into the review prompt
       |
       v
    [LLM]        -> ChatGroq writes the review comments as JSON
       |
       v
    [Parser]     -> turns the model's reply into plain text
       |
       v
    [Validator]  -> keeps only comments whose citation was really retrieved

LangChain concepts used here (good to be able to explain in the viva):
  - Document          : a piece of text + metadata (one function from the repo)
  - Retriever         : given a question/code, returns the most relevant Documents
  - ChatPromptTemplate: a prompt with {placeholders} filled in at run time
  - Chat model        : the LLM wrapper (ChatGroq)
  - Output parser     : converts the model's message into a Python value
  - Runnable / LCEL   : every step above can be chained with the | operator
"""

import os
import re
from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from codelens import embed_texts
from redact import redact
from llm_review import MODEL, SYSTEM_PROMPT, _norm, parse_and_validate


# ---------------------------------------------------------------- Retriever
class CodebaseRetriever(BaseRetriever):
    """LangChain retriever backed by our Chroma collection of code chunks."""

    collection: Any
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        res = self.collection.query(query_embeddings=embed_texts([query]), n_results=self.k)
        docs = []
        for chunk_id, code, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
            docs.append(Document(page_content=code, metadata={"id": _norm(chunk_id), **meta}))
        return docs


# ------------------------------------------------------------------ Prompt
def format_docs(docs: list[Document]) -> str:
    """Turn retrieved Documents into the REFERENCE block shown to the LLM."""
    return "\n\n".join(f"### REFERENCE ID: {d.metadata['id']}\n{d.page_content}" for d in docs)


# Curly braces in the system prompt (the JSON example) must be escaped, because
# ChatPromptTemplate treats {name} as a placeholder.
_SYSTEM_TEXT = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_TEXT),
        ("human", "{references}\n\n### NEW FUNCTION TO REVIEW\n{new_code}"),
    ]
)


# --------------------------------------------------------------------- LLMs
def get_llm():
    """The real model: Groq via LangChain, asked to reply in JSON."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it in your terminal first."
        )
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=MODEL, temperature=0.2)
    return llm.bind(response_format={"type": "json_object"})


def get_mock_llm():
    """Offline stand-in used ONLY to test the chain without an API key.

    Cites the first reference it sees (kept) and one made-up source (dropped).
    """
    import json

    def fake(prompt_value) -> str:
        text = prompt_value.to_string()
        first_id = re.search(r"### REFERENCE ID: (.+)", text).group(1).strip()
        return json.dumps(
            {
                "comments": [
                    {"issue": "Naming convention", "comment": "Uses camelCase; references use snake_case.", "cited_source": first_id},
                    {"issue": "Made-up citation", "comment": "Cites a function that does not exist.", "cited_source": "nowhere.py::imaginary"},
                ]
            }
        )

    return RunnableLambda(fake)


# -------------------------------------------------------------------- Chain
def build_chain(retriever: BaseRetriever, llm):
    """retriever | prompt | llm | parser | validator, as one LangChain chain.

    Input : {"new_code": "<source code to review>"}
    Output: {"new_code", "docs", "raw", "kept", "dropped", "redactions"}
    """

    def validate(state: dict) -> dict:
        valid_ids = [d.metadata["id"] for d in state["docs"]]
        kept, dropped = parse_and_validate(state["raw"], valid_ids)
        return {**state, "kept": kept, "dropped": dropped}

    def safe_inputs(s: dict) -> dict:
        """Redact secrets from BOTH the references and the new code before the LLM sees them."""
        return {
            "references": redact(format_docs(s["docs"]))[0],
            "new_code": redact(s["new_code"])[0],
        }

    review_step = (
        safe_inputs
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(docs=itemgetter("new_code") | retriever)  # 1. retrieve
        | RunnablePassthrough.assign(raw=review_step)                        # 2-4. prompt -> LLM -> parse
        | RunnableLambda(validate)                                           # 5. check citations
        | RunnablePassthrough.assign(                                        # 6. report what was redacted
            redactions=lambda s: redact(s["new_code"])[1]
            + [f for d in s["docs"] for f in redact(d.page_content)[1]]
        )
    )