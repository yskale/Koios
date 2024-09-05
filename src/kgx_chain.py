from fastapi import FastAPI
from typing import Any, List, Tuple
from pydantic import Field, BaseModel
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
from langchain_community.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langserve import add_routes
import pickle
import json
from operator import itemgetter
import config
from langchain import hub
import logging
logging.basicConfig(level=logging.INFO)
import requests
import pandas as pd
import redis
from redisgraph import Node, Edge, Graph, Path





app = FastAPI()



class Question(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})

template = """You are a professor at a prestigious university. 
You have information of about studies given to you as abstracts in the following format.

    Study name1 (study id1): 
    study description 1
    Variable name and its description

    Study name2 (study id2):
    study description  2 
    Variable name and its description

     
    ...

for eg:

    NHLBI TOPMed: Cleveland Clinic Atrial Fibrillation (CCAF) Study (phs001189): 
    
    The Cleveland Clinic Atrial Fibrillation Study consists of clinical and genetic data ....


 
Your task is to answer a user question based on the abstracts and variable details. 
Please include references using the provided abstracts in your answer. 
Your answers should be factual. Do not suggest anything that is not in the abstract information. 
If you can not find answer to the question please say there is not enough information to answer the question.
Respond with just the answer to the question, don't tell the user what your did. 
Don't INCLUDE the PHRASE "based on the provided abstracts.
Always include the study IDs in your answer.

Answer the question based only on the following information:

<context>
{con_context}
</context>
"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")



def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Load the graph from a .pkl file
def load_graph(graph_file):
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    return graph

        

# Retrieve studies based on matching concepts
def retrieve_studies(concepts,redis_graph):
    def concept_details_kgx(concept_id):
        #concept_name = concept_name.replace('"', '\\"')
        
        query = f"""
            MATCH (c {{id: "{concept_id}"}})-[r1]->(v:`biolink.StudyVariable`)-[r2]->(s:`biolink.Study`)
            RETURN c.name AS concept_name,v.name AS variable_name,v.id AS variable_id,v.description AS variable_desc, s.id AS study_id
            LIMIT 50
        """

        result = redis_graph.query(query)
        if result.result_set is None:
            return pd.DataFrame(columns=['concept_name', 'variable_name','variable_id','variable_desc','study_id'])
        data = result.result_set
        df = pd.DataFrame(data, columns=['concept_name', 'variable_name','variable_id','variable_desc','study_id'])
        df['study_id'] = df['study_id'].apply(lambda x: x.split('.')[0])
    
        return df

    def get_concept_identifier(concepts):
        concept_mapping = {}
        for concept in concepts:
            url = "https://sap-qdrant.apps.renci.org/annotate/"
            payload = {
                "text": concept,
                "model_name": "sapbert",
                "count": 1
            }
            response = requests.post(url, json=payload).json()
            if response:
                concept_mapping[concept] = response[0]['curie']
        return concept_mapping
    def get_study_details(concept_ids):
        dfs = []
        for concept in concept_ids.values():
            df = concept_details_kgx(concept)
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        df_summary = final_df.groupby('study_id').agg(
            concept_list=('concept_name', lambda x: ', '.join(x)),#.join(sorted(set(x)))),
            variable_desc_list=('variable_desc', lambda x: ', '.join(x)),
            variable_name_list=('variable_name', lambda x: ', '.join(x)),
            variable_id_list=('variable_id', lambda x: ', '.join(x)),
            number_of_concepts=('concept_name', 'nunique')
        ).reset_index()
        df_summary = df_summary.sort_values(by='number_of_concepts', ascending=False)
        df_summary['variable_id_list'] = df_summary['variable_id_list'].apply(
            lambda x: ', '.join(set([vid.split('.')[0] for vid in x.split(', ')]))
        )

        # Assuming get_study_data is defined as given
        def get_study_data(study_id):
            with open("/Users/ykale/Documents/Local-Dev/koios/Koios/prompts/bdc_121_studies.json") as stream:  # Replace with the path to your studies JSON file
                data = json.load(stream)
            for study in data:
                if study['StudyId'].split('.')[0] == study_id.split('.')[0]:
                    return {
                        "study_name": study['StudyName'],
                        "permalink": study['Permalink'],
                        "description": study['Description'],
                    }
            return {"study_name": None, "permalink": None, "description": None}
        df_summary[['study_name', 'permalink', 'description']] = df_summary['study_id'].apply(
            lambda x: pd.Series(get_study_data(x)))

        # Concatenate the `variable_name`, `variable_id`, and `variable_desc`
        df_summary['variable_info'] = df_summary.apply(
            lambda row: ', '.join([f"{name} ({var_id}): {desc}" 
                                for name, var_id, desc in zip(
                                    row['variable_name_list'].split(', '),
                                    row['variable_id_list'].split(', '),
                                    row['variable_desc_list'].split(', ')
                                )]), axis=1)

        # Display the specific columns
        df_display = df_summary[['study_id', 'description', 'variable_info','number_of_concepts']]
        top_studies = df_display.head(10)
   

        documents = []
        for _, row in top_studies.iterrows():
            study_data = get_study_data(row['study_id'])
            if study_data:
                # Create the document in dictionary format (JSON serializable)
                doc = {
                    "page_content": f"{row['description']}\n\n Variable_info: \n{row['variable_info']}",
                    "metadata": {
                        "study_id": row['study_id'],
                        "study_name": study_data['study_name'],
                        "permalink": study_data['permalink']
                    }
                }
                documents.append(doc)
        return  documents

    
    df_summary = get_study_details(get_concept_identifier(concepts))
    return  df_summary
    

# Functions to extract biomedical concepts
def extract_biomedical_concept(input_text: str, llm) -> str:
    prompt = f"Identify biomedical concepts from the query only and do not add addional terms. Here is the user query: {input_text}. The extracted biomedical concepts must be separated by commas and no text before or after it. Do not add if the concept is not biomedical"
    concept = llm.invoke(prompt)
    return concept

# Initialize the chain for concept
def init_concept_chain():
    llm = Ollama(
        base_url=config.OLLAMA_URL,
        model=config.GEN_MODEL_NAME
    )
    kgx = config.redis_graph
    redis_graph = Graph('test', kgx)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Define the steps
    extract_concept_step = RunnableLambda(lambda x: extract_biomedical_concept(x, llm))
    retrieve_studies_step = RunnableLambda(lambda concepts: retrieve_studies(concepts.split(", "),redis_graph))

    search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x['chat_history'])
            )
            | rephrase_prompt
            | llm
            | StrOutputParser()
        ),
        RunnableLambda(itemgetter("input")),
    )

    _inputs = RunnableParallel(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "con_context": search_query | extract_concept_step | retrieve_studies_step #| combine_documents_step,
        }
    ).with_types(input_type=Question)

    qachain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()

    return qachain

add_routes(
    app,
    init_concept_chain(),
    path='/dug-qa',
    input_type=Question
)

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="localhost", port=8000)
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")

