from typing import Optional
from dataclasses import dataclass, field

@dataclass
class PaperInfo:
    paper_id: str
    paper_name: str
    reference_key: Optional[str] = None
    authors: Optional[str] = None
    publisher: Optional[str] = None
    journal_name: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    publication_month_year: Optional[str] = None

@dataclass
class ChatQuery:
    query_str: str
    paper_ids: list[str]
    history: list = field(default_factory=list)
    summary: Optional[str] = None