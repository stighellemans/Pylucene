import lucene
import re
from typing import Optional, List, Tuple

from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, FuzzyQuery
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.io import StringReader

DocID = int

def query_database(index_directory: str, query_str: str, top_k=10, similarity: str="BM25", custom_analyzer: Optional[Analyzer]=None
                   ) -> List[Tuple[DocID, float]]:
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


def fuzzy_query_database(index_directory: str, query_str: str, top_k=10, similarity: str="BM25", custom_analyzer: Optional[Analyzer]=None, fuzziness: int=1
                                               ) -> List[Tuple[DocID, float]]:
    """Search for a fuzzy query in the index with analyzer processing and return the top k results."""
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

    # Step 1: Analyze the query string to get processed tokens
    stream = analyzer.tokenStream("", StringReader(query_str))
    char_term_attr = stream.addAttribute(CharTermAttribute.class_)  # Use CharTermAttribute.class_
    stream.reset()
    tokens = []
    while stream.incrementToken():
        tokens.append(char_term_attr.toString())  # Access each token's text
    stream.end()
    stream.close()

    # Step 2: Build a Boolean query with fuzzy matching for each token
    boolean_query = BooleanQuery.Builder()
    for token in tokens:
        term = Term("text_content", token)
        fuzzy_query = FuzzyQuery(term, fuzziness)  # Apply fuzziness level
        boolean_query.add(fuzzy_query, BooleanClause.Occur.SHOULD)

    combined_query = boolean_query.build()
    hits = searcher.search(combined_query, top_k).scoreDocs
    results = [(int(searcher.doc(hit.doc).get("doc_id")), hit.score) for hit in hits]

    reader.close()
    return results



if __name__ == "__main__":
    # intitialize VM to adapt java lucene to python
    lucene.initVM()

    index_dir = "/root/index/small"

    results = query_database(index_directory=index_dir, query_str="Milestones", top_k=5)
    print(results)