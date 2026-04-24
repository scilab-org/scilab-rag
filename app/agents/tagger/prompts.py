CHUNK_SUMMARY_PROMPT = """
Summarize this section of a research paper in 3-5 sentences.
Focus on methods, findings, and contributions only.
Do NOT include any Keywords or Index Terms line.

Section:
{context}
"""

GLOBAL_SUMMARY_PROMPT = """
You are consolidating section summaries of a research paper.
Create a single coherent summary of the entire paper.

Rules:
- Capture main research domain.
- Highlight key methods and contributions.
- Avoid repetition.
- Keep it under 250 words.
- Do NOT include any Keywords or Index Terms line.

Section summaries:
{context}
"""

TAG_FROM_SUMMARY_PROMPT = """
You are generating structured research keywords for a scientific paper.

TASK:
Based on the research paper summary, generate and extract research keywords
that include both domain-level and specific technical keywords.

EXISTING KEYWORDS (do not duplicate these):
{existing_tags}
OFFICIAL KEYWORDS (already verified from paper):
{official_keywords}

STEP 1: IDENTIFY OFFICIAL KEYWORDS
- If OFFICIAL KEYWORDS is not "None":
  - Include ALL of them exactly as provided — never truncate or omit any.
  - Mark every one as isFromPaper: true.
- If OFFICIAL KEYWORDS is "None":
  - DO NOT create or infer any official keywords.
  - All tags must have isFromPaper: false.

STEP 2: GENERATE ADDITIONAL KEYWORDS
1. Official keywords from Step 1 are fixed — include all of them regardless of count.
2. If official keywords total fewer than 6, generate enough additional keywords to bring the combined total to 6-10.
3. If official keywords already total 6 or more, generate 0-3 complementary keywords only if they add meaningful coverage not present in the official list.
4. Each generated keyword: 2-4 words, noun phrases only (no verbs, no sentences).
5. Do NOT repeat, expand, abbreviate, or rephrase any official or existing keyword.
6. Avoid semantic duplicates (abbreviations, singular/plural, synonymous phrases).
7. If existing keywords already sufficiently cover the summary, generate none.
8. If the summary has no scientific content, return {"tags": []}.
9. All generated keywords must have isFromPaper: false — always, without exception.

OUTPUT FORMAT — return ONLY valid JSON, no text before or after.
Official keywords (isFromPaper: true) appear first, then generated keywords:
{
  "tags": [
    {"name": "Keyword One", "isFromPaper": true},
    {"name": "Keyword Two", "isFromPaper": false}
  ]
}

Now generate keywords for:
{context}
"""