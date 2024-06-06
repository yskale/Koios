
import requests,json
from config import ollama_emb,QClient,llm_url



def query_embed(user_query,ollama_emb):
    #ollama_emb = OllamaEmbeddings(model="llama3:latest", base_url="http://localhost:11434")
    #phi3:14b-medium-128k-instruct-f16
    query_vector = ollama_emb.embed_query(user_query)
    return query_vector



def similar_questions(query_vector,QClient):
    #Qclient = QdrantClient(url="http://localhost:53814")
    topk_questions = QClient.search(
    collection_name="test_collection2",
    query_vector=query_vector,
    limit=15  # Return 5 closest points
    )
    return topk_questions



def study_id(topk_questions):
    study_ids = list()
    for pt in topk_questions:
        study_ids.append([pt.payload['question_id'].split('.')[0], pt.score])
    return study_ids



def lookup_study_abstract(study_ids):
    import json
    with open('/Users/ykale/Documents/Dev/koios/Koios/prompts/99_select_studies.json') as stream:
        studies_all = json.load(stream)
    selected_studies = {}
    
    for study in studies_all:
        for study_id, score in study_ids:
            if study['StudyName']  == study_id:
                selected_studies[study['StudyName']] = study
                selected_studies[study['StudyName']]['scores'] = selected_studies[study['StudyName']].get('scores',[])
                selected_studies[study['StudyName']]['scores'].append(score)
    return selected_studies



def retrive_the_study_abst(study_list):
    study_context=[]
    for study in study_list:
        study_context.append(study)
    return study_context



def studies_to_context(studies):
    context = ""
    for k, v in studies.items():
        #print(v)
        context += f"Study Name: {v['StudyName']}\n" + f"{v['Description']} \n\n"
    return context



def answer_user_query(user_query, context,llm_url): 
    import json
    prompt = f"""
    Answer this question: "{user_query}"    
    Please in the response, explain your reasoning step by step, and don't not infer out side this specific context. 
    Context: {context}
    """
    payload = {
        "model": "llama3:latest",
        "prompt": f"{prompt}",
        "stream": False,
        "temperature": 2,
        "seed":True}
    print(f"Asking LLM : {user_query}")
    response = requests.post(llm_url, json=payload)    
    return response.json()



def ask_question(user_query):
    print("\n",user_query)
    query_vector=query_embed(user_query,ollama_emb)
    print("Done1")
    topk_questions=similar_questions(query_vector,QClient)
    print("Done2")
    study_ids=study_id(topk_questions)
    print("Done3")
    studies=lookup_study_abstract(study_ids)
    print("Done4")
    context=studies_to_context(studies)
    print("Done5")
    #print(answer_user_query(user_query, context,llm_url)['response'])
    #fr=answer_user_query(user_query, context,llm_url)['response']
    return  answer_user_query(user_query, context,llm_url)['response']
