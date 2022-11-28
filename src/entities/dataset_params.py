from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetParams:
    root_path: str
    metadata: str
    years_list: List[int]
    split: Optional[str]
