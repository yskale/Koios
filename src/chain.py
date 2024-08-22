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
from langchain_core.output_parsers import StrOutputParser
from typing import Any, List, Tuple
from pydantic import Field, BaseModel
from langchain_community.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import json
import fastapi
from operator import itemgetter
from langserve import add_routes

app = fastapi.FastAPI()


class Question(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


# RAG answer synthesis prompt
template = """You are a professor at a prestigious university. You have performed a database lookup of study abstracts. 
Your task is to answer the question provided based on the abstracts provided from your search, and add references to your answer. 
Your answers should be factual. Do not suggest anything that is not in the database lookup. 
Answer the question based only on the following database lookup:
<database lookup>
{context}
</database lookup>

"""
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
    docs_seen = []
    doc_strings = []
    for document in docs:
        if document.metadata['study_id'] not in docs_seen:
            doc_strings.append(format_document(document, document_prompt))
            docs_seen.append(document.metadata['study_id'])
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
                return study['StudyName'] + ' : ' + '\n' + study['Description']
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
    ).as_retriever(search_kwargs={'k': 20})
    if config.LLM_SERVER_TYPE == "VLLM":
        llm = ChatOpenAI(
            api_key="EMPTY",
            base_url=config.LLM_URL,
            model=config.GEN_MODEL_NAME
            )

    elif config.LLM_SERVER_TYPE == "OLLAMA":
        llm = Ollama(
            base_url=config.LLM_URL,
            model=config.GEN_MODEL_NAME
        )

    # see https://smith.langchain.com/hub/langchain-ai/chat-langchain-rephrase
    rephrase_prompt = PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {input}\nStandalone Question:")

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