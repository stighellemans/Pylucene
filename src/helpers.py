from typing import Dict, List

import numpy as np
import pandas as pd

QueryID = int
DocID = int


def process_query_results(
    queries: Dict[QueryID, str],
    query_results: pd.DataFrame,
    doc_col_name: str = "doc_number",
) -> Dict[QueryID, List[DocID]]:
    new_query_results = {}
    for q_id in queries:
        if q_id in query_results.index:
            relevant_docs = query_results.loc[q_id][doc_col_name]
            # Handle multiple relevant docs scenario
            if not isinstance(relevant_docs, np.int64):
                new_query_results[q_id] = [
                    int(doc_id) for doc_id in list(relevant_docs)
                ]  # If multiple relevant docs
            else:
                new_query_results[q_id] = [int(relevant_docs)]
    return new_query_results
