KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
You are extracting knowledge graph entities and relationships from a CHUNK of a scientific research paper.
Multiple chunks are processed independently and merged later.
Extract up to {max_knowledge_triplets} triplets clearly supported by the chunk.

## STRUCTURAL ANCHOR
Every chunk must begin with exactly these two entities and one relationship:
1. A Paper entity with entity_name = "{paper_title}"
2. A Section entity with entity_name = "{section_headings}"
3. A HAS_SECTION relationship from the Paper to the Section

## PAPER CONTEXT
Paper Title: {paper_title}
Section: {section_headings}

## ENTITY TYPES
Use the categories below as guidance. Pick the best fit. If no listed type fits well,
introduce a short, clear PascalCase type (e.g. "Biomarker", "PolicyIntervention").
Use the "Entity" fallback only as a last resort.

  Universal   : Paper, Author, Institution, Venue
  Scientific  : Concept, Theory, Hypothesis, Claim, Variable, Factor
  Methods     : Method, Model, Algorithm, Framework, Protocol, Intervention
  Artifacts   : Tool, Software, Dataset, Instrument, Benchmark, Scale
  Outcomes    : Finding, Result, Metric, Statistic, Evidence
  Structural  : Section, Table, Figure, Formula

## RELATIONSHIP VOCABULARY
Prefer a label from this list. If none fits precisely, use a clear SCREAMING_SNAKE_CASE
verb phrase (e.g. MODERATES, MEDIATES, PREDICTS, INTERACTS_WITH).
Always use RELATED_TO when no more specific label applies.

  Causal        : CAUSES, LEADS_TO, PREVENTS, REDUCES, INCREASES, INFLUENCES,
                  MODERATES, MEDIATES, PREDICTS, EXACERBATES, MITIGATES
  Evaluative    : EVALUATES, COMPARES, OUTPERFORMS, MEASURES, VALIDATES, SUPPORTS,
                  CONTRADICTS, REPLICATES
  Methodological: USES, APPLIES, IMPLEMENTS, EXTENDS, BASED_ON, ADAPTED_FROM
  Descriptive   : DEFINES, DESCRIBES, PRESENTS, MENTIONS, CHARACTERIZED_BY
  Associative   : ASSOCIATED_WITH, CORRELATED_WITH, RELATED_TO
  Structural    : HAS_SECTION, PART_OF, CITES, WRITTEN_BY, PUBLISHED_IN

## EXTRACTION RULES
1. Always emit the Paper → Section anchor triplet first (see STRUCTURAL ANCHOR).
2. Extract entities and relationships clearly supported by the chunk, including
   those strongly implied by context. Do not hallucinate.
3. "{paper_title}" is the source paper. Use this exact title if the chunk says
   "this study", "we found", "the present paper", etc.
4. Use canonical, concise names: "Social Media" not "the use of social media platforms".
5. Every relationship must reference entities that appear in your entities list.
6. Capture nuance in entity_description and relationship_description — not in the type.
7. Fewer than {max_knowledge_triplets} triplets is fine.
   Return empty lists only if the chunk is truly content-free (e.g. a reference list).
8. Relation labels: SCREAMING_SNAKE_CASE, no spaces, no punctuation.
   Source and target entity names must match exactly.
9. Before finalising, do a quick self-check: are there important entities or
   relationships present in the chunk that you have not yet captured? Add them now.

## OUTPUT FORMAT
Valid JSON only. No markdown. No extra text. Start with {{ end with }}.
If nothing extractable: {{ "entities": [], "relationships": [] }}

Required keys:
- Each entity   : "entity_name", "entity_type", "entity_description"
- Each relation : "source_entity", "target_entity", "relation", "relationship_description"

## EXAMPLES

### Example 1 — CS / ML paper

Input chunk (paper: "GraphRAG: Community-Aware Retrieval", section: "3. Methods"):
"We propose GraphRAG, a retrieval-augmented generation framework that uses
hierarchical Leiden clustering to detect communities in a Neo4j property graph.
The method is evaluated on the BEIR benchmark dataset and extends Dense Passage Retrieval."

Output:
{{
  "entities": [
    {{"entity_name": "GraphRAG: Community-Aware Retrieval", "entity_type": "Paper", "entity_description": "The source paper proposing GraphRAG"}},
    {{"entity_name": "3. Methods", "entity_type": "Section", "entity_description": "The methods section describing the GraphRAG framework design"}},
    {{"entity_name": "GraphRAG", "entity_type": "Framework", "entity_description": "A retrieval-augmented generation framework that uses community detection over property graphs"}},
    {{"entity_name": "Hierarchical Leiden Clustering", "entity_type": "Algorithm", "entity_description": "A graph partitioning algorithm used to detect communities at multiple granularities"}},
    {{"entity_name": "Neo4j", "entity_type": "Tool", "entity_description": "A property graph database used to store and query the knowledge graph"}},
    {{"entity_name": "BEIR", "entity_type": "Benchmark", "entity_description": "A heterogeneous benchmark suite for information retrieval evaluation"}},
    {{"entity_name": "Dense Passage Retrieval", "entity_type": "Method", "entity_description": "A dense retrieval baseline that GraphRAG extends"}}
  ],
  "relationships": [
    {{"source_entity": "GraphRAG: Community-Aware Retrieval", "target_entity": "3. Methods", "relation": "HAS_SECTION", "relationship_description": "This chunk belongs to the Methods section of the paper"}},
    {{"source_entity": "GraphRAG", "target_entity": "3. Methods", "relation": "PART_OF", "relationship_description": "GraphRAG is the primary contribution described in this section"}},
    {{"source_entity": "GraphRAG", "target_entity": "Hierarchical Leiden Clustering", "relation": "BASED_ON", "relationship_description": "GraphRAG uses Leiden clustering as its community detection backbone"}},
    {{"source_entity": "GraphRAG", "target_entity": "Neo4j", "relation": "IMPLEMENTS", "relationship_description": "The property graph is stored and queried in Neo4j"}},
    {{"source_entity": "GraphRAG", "target_entity": "BEIR", "relation": "EVALUATES", "relationship_description": "GraphRAG retrieval quality is measured against the BEIR benchmark"}},
    {{"source_entity": "GraphRAG", "target_entity": "Dense Passage Retrieval", "relation": "EXTENDS", "relationship_description": "GraphRAG builds upon and improves Dense Passage Retrieval"}}
  ]
}}

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