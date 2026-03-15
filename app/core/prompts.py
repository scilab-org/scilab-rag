"""
Prompt templates for knowledge graph extraction and query processing.
"""

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
You are extracting knowledge graph entities and relationships from a CHUNK of a scientific research paper.
Multiple chunks are processed independently and merged later.
Extract up to {max_knowledge_triplets} triplets explicitly supported by the chunk.

## SECTION CONTEXT
This chunk comes from the following section(s) of the paper:
{section_headings}

## PAPER CONTEXT
Paper Title: {paper_title}

## ENTITY TYPES
Allowed ENTITY TYPES (exactly these, no subtypes):
- Paper     : a research paper (source or cited)
- Author    : a person who authored a paper
- Section   : a named structural part of the paper (e.g., "Introduction", "Results")
- Concept   : a technical idea, theory, principle, or domain term
- Method    : an algorithm, model, technique, or approach
- Tool      : a software, instrument, device, or system used in the work
- Dataset   : a dataset, corpus, cohort, specimen collection, or experimental data source
- Finding   : a result, conclusion, observation, or measured outcome
- Reference : a bibliographic citation

## ALLOWED RELATIONSHIPS
The "relation" field MUST be copied VERBATIM from this list. 
Any other value is a critical error.

Paper-centered:
- Paper    -> Section   : HAS_SECTION
- Author   -> Paper     : WROTE

Structural:
- Section  -> Concept   : MENTIONS
- Section  -> Method    : DESCRIBES
- Section  -> Finding   : PRESENTS
- Section  -> Tool      : MENTIONS
- Section  -> Dataset   : MENTIONS

Technical:
- Method   -> Concept   : BASED_ON
- Method   -> Tool      : IMPLEMENTED_IN
- Method   -> Dataset   : EVALUATED_ON
- Method   -> Method    : EXTENDS
- Tool     -> Tool      : INTEGRATES
- Finding  -> Method    : PRODUCED_BY
- Concept  -> Concept   : RELATED_TO

CRITICAL RULES FOR RELATIONSHIPS:
- If the correct relation is not in the list above → DROP the relationship entirely.
- NEVER invent a relation string. NEVER use lowercase. NEVER use spaces.
- "source_entity" and "target_entity" MUST exactly match an "entity_name" from your entities list.
- The (source_type → target_type) pair MUST match the table above.
  e.g. Method -> Dataset : EVALUATED_ON is valid.
       Dataset -> Method : EVALUATED_ON is NOT valid.

## EXTRACTION RULES
1. Extract ONLY entities and relationships explicitly stated in the chunk.
2. Do NOT invent entities or relationships.
3. "{paper_title}" is the SOURCE paper. If the chunk says "this paper" / "we propose", 
   use this exact title as the Paper entity name.
4. Do NOT force a Paper entity if the chunk has none.
5. Use consistent canonical names across chunks (e.g., "DocLayNet" not "the DocLayNet dataset").
6. Capture nuance in entity_description, not entity_type.
7. Fewer than {max_knowledge_triplets} triplets is fine. 
   Return empty lists if nothing is extractable.
8. Each relationship must reference valid entities from your entities list.

## OUTPUT FORMAT
Valid JSON only. No markdown. No extra text. Start with {{ end with }}.
If nothing extractable: {{ "entities": [], "relationships": [] }}

Required keys:
- Each entity   : "entity_name", "entity_type", "entity_description"
- Each relation : "source_entity", "target_entity", "relation", "relationship_description"

## CORRECT EXAMPLE

Input chunk:
"We propose GraphRAG, a retrieval-augmented generation framework that uses 
hierarchical Leiden clustering to detect communities in a Neo4j property graph. 
The method is evaluated on the BEIR benchmark dataset and extends Dense Passage Retrieval."

Output:
{{
  "entities": [
    {{"entity_name": "GraphRAG", "entity_type": "Method", "entity_description": "A retrieval-augmented generation framework using community detection over property graphs"}},
    {{"entity_name": "Hierarchical Leiden Clustering", "entity_type": "Method", "entity_description": "A graph partitioning algorithm used to detect communities at multiple granularities"}},
    {{"entity_name": "Neo4j", "entity_type": "Tool", "entity_description": "A property graph database used to store and query the knowledge graph"}},
    {{"entity_name": "BEIR", "entity_type": "Dataset", "entity_description": "A heterogeneous benchmark for information retrieval evaluation"}},
    {{"entity_name": "Dense Passage Retrieval", "entity_type": "Method", "entity_description": "A dense retrieval baseline that GraphRAG extends"}}
  ],
  "relationships": [
    {{"source_entity": "GraphRAG", "target_entity": "Hierarchical Leiden Clustering", "relation": "BASED_ON", "relationship_description": "GraphRAG uses Leiden clustering as its community detection backbone"}},
    {{"source_entity": "GraphRAG", "target_entity": "Neo4j", "relation": "IMPLEMENTED_IN", "relationship_description": "The property graph is stored and queried using Neo4j"}},
    {{"source_entity": "GraphRAG", "target_entity": "BEIR", "relation": "EVALUATED_ON", "relationship_description": "GraphRAG's retrieval quality is measured against the BEIR benchmark"}},
    {{"source_entity": "GraphRAG", "target_entity": "Dense Passage Retrieval", "relation": "EXTENDS", "relationship_description": "GraphRAG builds upon and improves over Dense Passage Retrieval"}}
  ]
}}

## WRONG EXAMPLE (never do this)
{{
  "relationships": [
    {{"source_entity": "GraphRAG", "target_entity": "BEIR", "relation": "tested_on"}},
    {{"source_entity": "the proposed method", "target_entity": "BEIR", "relation": "EVALUATED_ON"}},
    {{"source_entity": "BEIR", "target_entity": "GraphRAG", "relation": "EVALUATED_ON"}}
  ]
}}
Reasons: "tested_on" not in allowed list. "the proposed method" not in entities list. Direction is wrong.

## CHUNK TEXT
{text}

output:
"""

GRAPH_QA_SYSTEM_PROMPT = """\
You are HyperDataLab Assistant, a scientific research assistant built by \
the HyperDataLab team. You help researchers understand, compare, and \
synthesize information from their uploaded scientific papers.

## Behavioral rules
1. Answer ONLY from the information retrieved from the user's papers \
   (provided below as "Paper Notes"). If the notes do not contain enough \
   information to answer, say so honestly — never fabricate.
2. NEVER mention internal implementation details such as "knowledge graph", \
   "entities", "relationships", "triplets", "nodes", "edges", "embeddings", \
   "vector search", or "retrieval". From the user's perspective you simply \
   read their papers.
3. Cite papers by title when the information comes from a specific paper.
4. When multiple papers discuss the same topic, synthesize and compare \
   rather than repeating each paper separately.
5. Be concise, precise, and use academic tone. Use bullet points or \
   numbered lists when appropriate.
6. If asked about something unrelated to the user's papers, politely \
   redirect: "I can only help with questions about your uploaded papers."
"""

GRAPH_QA_USER_PROMPT = """\
## Paper Notes
{context}

## Question
{question}
""" 


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

IMAGE_DESCRIPTION_PROMPT = """
Write a single formal academic paragraph describing the image. 
Describe observable visual elements and their apparent scientific role or function. 
Allow concise explanation of relationships, processes, or methodology inferred directly from the visual content. 
Use impersonal research-manuscript language. 
Avoid conversational or meta statements.
"""

FORMULA_DESCRIPTION_PROMPT = """
Write a concise academic paragraph describing the mathematical expressions and code visible in the image. 
Explain their structural components and apparent purpose based solely on the presented notation. 
Use formal, impersonal research style. 
Do not include conversational or meta language.
"""