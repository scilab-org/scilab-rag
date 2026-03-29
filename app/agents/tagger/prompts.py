CHUNK_SUMMARY_PROMPT = """
You are summarizing a section of a research paper.

Summarize the following text clearly and concisely.

Rules:
- Focus on main methods, contributions, and technical ideas.
- Ignore minor implementation details.
- Do not invent information.
- Write 5-8 sentences maximum.

Text:
{context}
"""

GLOBAL_SUMMARY_PROMPT = """
You are consolidating section summaries of a research paper.

Create a single coherent summary of the entire paper.

Rules:
- Capture main research domain.
- Highlight key methods and contributions.
- Avoid repetition.
- Keep it under 200 words.

Section summaries:
{context}
"""

TAG_FROM_SUMMARY_PROMPT = """
You are generating structured research tags for a scientific paper.

TASK:
Based on the research paper summary, generate 6-10 research tags
that include BOTH generic (domain-level) and specific (method-level) tags.

Some tags may already exist. Do NOT generate duplicates
or semantically similar tags.

EXISTING TAGS:
{existing_tags}

REQUIREMENTS:

TAG MIX RULES:
1. Generate:
   - 2-3 Domain-Level tags (broad research area).
   - 3-6 Specific Technical tags (methods, techniques, models, algorithms, applications).

2. Domain-Level tags:
   - 2-3 words.
   - Represent research field or subfield.
   - Examples: "Graph Machine Learning", "Natural Language Processing", "Computer Vision".

3. Specific Technical tags:
   - 2-4 words.
   - Represent core methodology, model, algorithm, framework, or application.
   - Must be more specific than domain tags.

GENERAL RULES:
4. Use noun phrases only (no verbs, no sentences).
5. Each tag must contain 2-4 words.
6. Do NOT repeat, expand, abbreviate, or rephrase any existing tags.
   (e.g., if "LLM" exists, do NOT generate "Large Language Models")
7. Avoid semantic duplicates, including:
   - Abbreviations vs full names
   - Singular vs plural
   - Closely synonymous phrases
8. If existing tags already sufficiently cover the summary,
   return an empty list.
   
9. If the summary:
- Is meaningless,
- Contains random characters,
- Contains no scientific content,
- Or is too short to determine a research domain,

Return:
{
    "tags": []
}

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:

{
  "tags": [
    "Tag One",
    "Tag Two"
  ]
}

Do not include any text before or after the JSON.

----------------------------------------
EXAMPLE

Existing Tags:
["Knowledge Graph", "Semantic Search"]

Input Summary:
"This paper proposes a graph-based retrieval-augmented generation framework
using hierarchical Leiden clustering, property graph indexing,
and contrastive embedding alignment for large-scale scientific corpora."

Expected Output:
{
  "tags": [
    "Knowledge Graph",
    "Semantic Search",
    "Graph Machine Learning",
    "Information Retrieval",
    "Retrieval-Augmented Generation",
    "Hierarchical Leiden Clustering",
    "Property Graph Indexing",
    "Contrastive Embedding Alignment"
  ]
}

----------------------------------------

Now generate tags for the following summary:

{context}
"""