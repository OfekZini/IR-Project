from typing import List, Tuple


DocId = int
Tf = int
Score = float

Posting = Tuple[DocId, Tf]
RankedPosting = Tuple[DocId, Score]

PostingList = List[Posting]
RankedPostingList = List[RankedPosting]
Tokens = List[str]