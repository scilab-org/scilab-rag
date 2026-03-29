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
- Section -> Table   : PRESENTS
- Section -> Formula : DEFINES

Technical:
- Method   -> Concept   : BASED_ON
- Method   -> Tool      : IMPLEMENTED_IN
- Method   -> Dataset   : EVALUATED_ON
- Method   -> Method    : EXTENDS
- Tool     -> Tool      : INTEGRATES
- Finding  -> Method    : PRODUCED_BY
- Concept  -> Concept   : RELATED_TO
- Method  -> Formula : FORMALIZED_AS
- Finding -> Table   : SUMMARIZED_IN

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