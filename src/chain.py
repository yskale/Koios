import config
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from typing import Any
from langchain_community.docstore.document import Document
from langchain_community.llms import Ollama
from langserve import CustomUserType
import json
import fastapi
from langserve import add_routes

app = fastapi.FastAPI()

class Question(CustomUserType):
    question: str





class CustomQdrant(Qdrant):
    @classmethod
    def get_study_data(cls, study_id):
        with open(config.STUDIES_JSON_FILE) as stream:
            data = json.load(stream)
        for study in data:
            if study['StudyName'] == study_id:
                return study['StudyName'] + '(' + study['Permalink'] + '): ' + '\n' + study['Description']
        return ""

    @classmethod
    def _document_from_scored_point(
            cls,
            scored_point: Any,
            collection_name: str,
            content_payload_key: str,
            metadata_payload_key: str,
    ) -> Document:
        '''
        This method is overriden to get the documents form a local file to provide for the context.
        :param scored_point:
        :param collection_name:
        :param content_payload_key:
        :param metadata_payload_key:
        :return:
        '''
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        study_id = scored_point.payload.get('question_id').split('.')[0]
        metadata["study_id"] = study_id
        metadata["score"] = scored_point.score
        page_content = CustomQdrant.get_study_data(study_id)
        return Document(
            page_content=page_content,
            metadata=metadata,
        )

def init_chain():
    client = config.AQClient
    embeddings = config.ollama_emb
    doc_store = CustomQdrant(
        async_client=config.AQClient,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
        client=config.QClient
    )
    ollama = Ollama(
        base_url=config.OLLAMA_URL,
        model=config.GEN_MODEL_NAME
    )
    qachain = RetrievalQA.from_chain_type(ollama, retriever=doc_store.as_retriever(
        search_type="similarity_score_threshold",
        # I don't think this is working
        search_kwargs={'score_threshold': 0.6, 'k': 15}
    ), ).with_types(input_type=Question)

    return qachain

add_routes(
    app,
    init_chain(),
    path='/dug-qa',
    input_type=Question
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)