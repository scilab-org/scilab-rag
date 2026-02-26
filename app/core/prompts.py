"""
Prompt templates for knowledge graph extraction and query processing.
"""

KG_TRIPLET_EXTRACT_TMPL = """
-Context-
The input is a CHUNK of a larger document.
All chunks belong to ONE and ONLY ONE Paper.
The Paper identity MUST remain consistent across chunks.

-Goal-
Extract a schema-compliant knowledge graph centered on exactly ONE Paper entity.
All entities must be connected (directly or indirectly) to this Paper.

Extract up to {max_knowledge_triplets} triplets.

========================
KNOWLEDGE GRAPH SCHEMA
========================

Allowed ENTITY TYPES (only these):
- Paper
- Reference
- Section
- Argument
- Claim
- Evidence
- Concept
- Background
- Author

Entity Rules:
- Subtypes are NOT allowed.
- Do NOT encode subtype information in entity_type.
- Capture subtype details only in entity_description.
- Do NOT extract isolated entities.

========================
RELATIONSHIP RULES
========================

All relationships MUST:

1. Use ONLY the allowed relationship types below.
2. Be lowercase.
3. Use snake_case.
4. Contain no spaces.
5. Match EXACTLY one allowed relationship.
6. Follow the specified direction.

Allowed Relationships:

Paper-centered:
- Paper -> Section : has
- Paper -> Reference : cites
- Paper -> Background : has
- Paper -> Paper : cites
- Author -> Paper : wrote

Structural:
- Section -> Concept : mentions
- Section -> Argument : contains
- Argument -> Claim : has
- Claim -> Evidence : supported_by

Concept:
- Concept -> Concept : related_to

If a relationship does not exactly match the allowed list, DO NOT extract it.

========================
IDENTITY CONSTRAINTS
========================

- Exactly ONE Paper entity MUST exist.
- If the title appears, use it verbatim as entity_name.
- Otherwise infer a stable Paper identity.
- The same Paper name MUST be reused consistently.
- Every entity must connect to the Paper.
- No disconnected subgraphs.

========================
OUTPUT FORMAT
========================

Return valid JSON with EXACTLY two keys:
- "entities"
- "relationships"

The output:
- MUST start with '{' and end with '}'
- MUST contain no markdown fences
- MUST contain no extra text
- MUST be valid JSON

If nothing valid is found, return:
{ "entities": [], "relationships": [] }

========================
SAMPLE OUTPUT
========================

{
  "entities": [
    {
      "entity_name": "Attention Is All You Need",
      "entity_type": "Paper",
      "entity_description": "The main research paper introducing the Transformer architecture."
    },
    {
      "entity_name": "Introduction",
      "entity_type": "Section",
      "entity_description": "The section introducing the motivation of the paper."
    },
    {
      "entity_name": "Transformer Model",
      "entity_type": "Concept",
      "entity_description": "A neural architecture based on self-attention."
    },
    {
      "entity_name": "Vaswani et al. 2017",
      "entity_type": "Reference",
      "entity_description": "The cited work that introduced the Transformer."
    }
  ],
  "relationships": [
    {
      "source_entity": "Attention Is All You Need",
      "target_entity": "Introduction",
      "relation": "has",
      "relationship_description": "The paper contains an introduction section."
    },
    {
      "source_entity": "Introduction",
      "target_entity": "Transformer Model",
      "relation": "mentions",
      "relationship_description": "The introduction mentions the Transformer model."
    },
    {
      "source_entity": "Attention Is All You Need",
      "target_entity": "Vaswani et al. 2017",
      "relation": "cites",
      "relationship_description": "The paper cites the original Transformer work."
    }
  ]
}

========================
REAL DATA
========================
text: {text}

output:
"""

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """
You are provided with a set of relationships from a knowledge graph, each represented as 
entity1->entity2->relation->relationship_description. Your task is to create a summary of these 
relationships. The summary should include the names of the entities involved and a concise synthesis 
of the relationship descriptions. The goal is to capture the most critical and relevant details that 
highlight the nature and significance of each relationship. Ensure that the summary is coherent and 
integrates the information in a way that emphasizes the key aspects of the relationships.
"""

QUERY_ANSWER_PROMPT = """
Answer the question below as if you are responding directly to a user.

Guidelines:
- Do NOT mention or refer to any internal processes, summaries, or intermediate data.
- Do NOT use phrases such as "community summary", "based on the information above",
  "the provided data", "no community", or similar meta expressions.
- Provide a natural, confident answer.
- Reasoning and inference are allowed, but must remain implicit.

Question:
{query}
"""

AGGREGATE_ANSWERS_PROMPT = """
You are responding to a user question based on user given context.

Instructions:
- Produce a single final answer.
- Do NOT mention combining, aggregating, or synthesizing.
- Do NOT refer to previous answers or internal steps.
- The response must read as a direct standalone answer.
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
    "Graph Machine Learning",
    "Information Retrieval",
    "Retrieval-Augmented Generation",
    "Hierarchical Leiden Clustering",
    "Property Graph Indexing",
    "Contrastive Embedding Alignment"
  ]
}

In this example:
- "Graph Machine Learning" and "Information Retrieval" are Domain-Level tags.
- The remaining tags are Specific Technical tags.

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