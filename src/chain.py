import config
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any, List, Tuple
from pydantic import Field, BaseModel
from langchain_community.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
import json
import fastapi
from operator import itemgetter
from langserve import add_routes
from langchain import hub

app = fastapi.FastAPI()

class Question(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


# RAG answer synthesis prompt
template = """Answer the question based only on the following context:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer



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
    retriever = CustomQdrant(
        async_client=config.AQClient,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
        client=config.QClient
    ).as_retriever()
    llm = Ollama(
        base_url=config.OLLAMA_URL,
        model=config.GEN_MODEL_NAME
    )
    # see https://smith.langchain.com/hub/langchain-ai/chat-langchain-rephrase
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    search_query = RunnableBranch(
        # check history
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            # if there is some history reformat it and rephrase it and pass it down the chain
            RunnablePassthrough.assign(
                chat_history= lambda x: _format_chat_history(x['chat_history'])
        )
        | rephrase_prompt
        | llm
        | StrOutputParser()
        ),
        # no chat history , pass the whole question
        RunnableLambda(itemgetter("input")),
    )

    # lets bring this branch and the whole thing together

    _inputs = RunnableParallel(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "context": search_query | retriever | _combine_documents,
        }
    ).with_types(input_type=Question)

    qachain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()

    qachain = config.configure_langfuse(qachain)

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