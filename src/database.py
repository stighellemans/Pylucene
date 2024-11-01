import os
from tqdm import tqdm
import shutil
import time
from typing import Dict, Optional

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader, IndexOptions
from org.apache.lucene.document import Document, TextField, Field, StoredField, FieldType
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity


QueryID = int
DocID = int


def index_txt_file(ind_writer: IndexWriter, file_path: str, doc_id: DocID, store_original:bool=True):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_to_index = f.read()
    except UnicodeDecodeError:
        print(f"Skipping file {file_path} due to encoding issues.")
        return
    
    # choose if you want to store the original text to display alongside search results
    if store_original:
        store = Field.Store.YES
    else:
        store = Field.Store.NO

    doc = Document()
    doc.add(TextField("text_content", text_to_index, store))
    doc.add(StoredField("doc_id", doc_id))    # For retrieval

    ind_writer.addDocument(doc)

def make_database(doc_paths: Dict[DocID, str], index_directory: str, custom_analyzer: Optional[Analyzer]=None, 
                  similarity:str="BM25", store_original:bool=True, batch_size: int=5000) -> None:
    if custom_analyzer is None:
        analyzer = StandardAnalyzer()
    else:
        analyzer = custom_analyzer
    config = IndexWriterConfig(analyzer)

    if similarity == "BM25":
        config.setSimilarity(BM25Similarity())
    elif similarity == "tf-idf":
        config.setSimilarity(ClassicSimilarity())
    else:
        raise ValueError(f"{similarity} similarity not implemented.")
    
    # Ensure the index directory exists + remove old indices
    if os.path.exists(index_directory):
        shutil.rmtree(index_directory)
    os.makedirs(index_directory)
    index_dir = FSDirectory.open(Paths.get(index_directory))
    
    writer = IndexWriter(index_dir, config)
    
    try:
        for i, (doc_id, file_path) in enumerate(tqdm(list(doc_paths.items()), desc="Indexing documents")):
            index_txt_file(writer, file_path, doc_id, store_original)
            if (i + 1) % batch_size == 0:
                writer.commit()
    finally:
        writer.close()


def get_vocabulary(index_dir, field_name="text_content", max_terms=None):
    """
    Retrieve all unique terms for a specified field in a Lucene index.

    Parameters:
    index_dir (str): The directory path of the Lucene index.
    field_name (str): The name of the field from which to extract terms.
    max_terms (int, optional): Maximum number of terms to output. If None, outputs all terms.

    Returns:
    list: A list of terms from the specified field in the index.
    """
    # Open the directory and the index reader
    directory = FSDirectory.open(Paths.get(index_dir))
    reader = DirectoryReader.open(directory)
    vocabulary = []

    try:
        # Loop through each leaf (segment) in the index
        for leaf in reader.leaves():
            leaf_reader = leaf.reader()
            
            # Get terms for the specific field in this segment
            terms = leaf_reader.terms(field_name)
            if terms is not None:
                terms_enum = terms.iterator()  # Get the terms iterator
                # Collect terms into the vocabulary list
                for term in BytesRefIterator.cast_(terms_enum):
                    vocabulary.append(term.utf8ToString())
                    if max_terms and len(vocabulary) >= max_terms:
                        break
            if max_terms and len(vocabulary) >= max_terms:
                break
    finally:
        # Close the reader
        reader.close()
    
    return vocabulary


if __name__ == "__main__":
    # intitialize VM to adapt java lucene to python
    lucene.initVM()

    data_dir = "/root/data/full_docs_small"
    index_dir = "/root/index/small"

    start_time = time.time()
    make_database(data_directory=data_dir, index_directory=index_dir)
    end_time = time.time()

    vocab_size = len(get_vocabulary(index_dir=index_dir))
    print(f"Done after: {end_time - start_time:.2f}s with vocab size: {vocab_size}")