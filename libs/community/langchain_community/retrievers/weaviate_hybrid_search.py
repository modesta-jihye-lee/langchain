from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator


class WeaviateHybridSearchRetriever(BaseRetriever):
    """`Weaviate hybrid search` retriever.

    See the documentation:
      https://weaviate.io/blog/hybrid-search-explained
    """

    client: Any = None
    """keyword arguments to pass to the Weaviate client."""
    index_name: str
    """The name of the index to use."""
    text_key: str
    """The name of the text key to use."""
    alpha: float = 0.5
    """The weight of the text key in the hybrid search."""
    k: int = 4
    """The number of results to return."""
    embedding: Embeddings
    """Custom embedding models to use."""


    @root_validator(pre=True)
    def validate_client(
        cls,
        values: Dict[str, Any],
    ) -> Any:
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(values["client"], weaviate.WeaviateClient):
            client = values["client"]
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )

        schema = {
                    "class": values["index_name"],
                    "properties": [
                        {
                            "name": values["text_key"],
                            "dataType": ["text"],
                        }
                    ],
                }

        if not values["client"].collections.exists(values["index_name"]):
            values["client"].collections.create_from_dict(schema)

        return values

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        hybrid_search_kwargs: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        """Look up similar documents in Weaviate.

        query: The query to search for relevant documents
         of using weviate hybrid search.

        To use hybrid search with different options, please refer to
        https://weaviate.io/developers/weaviate/search/hybrid
        """

        query_obj = self.client.collections.get(self.index_name)
        query_vector = self.embedding.embed_query(query)

        if hybrid_search_kwargs is None:
            hybrid_search_kwargs = {}

        result = query_obj.query.hybrid(
            query=query,
            limit=self.k,
            alpha=self.alpha,
            vector=query_vector,
            **hybrid_search_kwargs,
            )

        docs = []

        for o in result.objects:
            text = o.properties.pop(self.text_key)
            docs.append(Document(page_content=text, metadata=o.properties))
        return docs
