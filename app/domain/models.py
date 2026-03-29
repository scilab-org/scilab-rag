from typing import Optional
from dataclasses import dataclass, field

@dataclass
class PaperInfo:
    paper_id: str
    paper_name: str

@dataclass
class ChatQuery:
    query_str: str
    paper_ids: list[str]
    history: list = field(default_factory=list)
    summary: Optional[str] = None