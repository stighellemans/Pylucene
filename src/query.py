import lucene
import re
from typing import Optional

from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity

def query_database(index_directory: str, query_str: str, top_k=10, similarity: str="BM25", custom_analyzer: Optional[Analyzer]=None):
    """Search for a query in the index and return the top k results."""
    if custom_analyzer is None:
        analyzer = StandardAnalyzer()
    else:
        analyzer = custom_analyzer

    # Open the index and set up the searcher
    index_dir = FSDirectory.open(Paths.get(index_directory))
    reader = DirectoryReader.open(index_dir)
    searcher = IndexSearcher(reader)

    # Set the similarity measure to match the indexing phase
    if similarity == "BM25":
        searcher.setSimilarity(BM25Similarity())
    elif similarity == "tf-idf":
        searcher.setSimilarity(ClassicSimilarity())
    else:
        raise ValueError(f"{similarity} similarity not implemented.")

    # Process the query
    query_parser = QueryParser("text_content", analyzer)
    query_str = re.sub(r'[*?[\]{}/:()^]+', '', query_str)  # Preprocess query to remove special symbols

    try:
        query = query_parser.parse(query_str)
        hits = searcher.search(query, top_k).scoreDocs
        results = [(int(searcher.doc(hit.doc).get("doc_id")), hit.score) for hit in hits]
    except Exception as e:
        print(f"Query parsing failed for '{query_str}': {e}")
        results = []  # Return an empty result for this query

    reader.close()
    return results


if __name__ == "__main__":
    # intitialize VM to adapt java lucene to python
    lucene.initVM()

    index_dir = "/root/index/small"

    results = query_database(index_directory=index_dir, query_str="Milestones", top_k=5)
    print(results)