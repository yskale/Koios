from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage
import json, re, os


chat_model = ChatOllama(model="llama3:latest", base_url="http://localhost:11434",temperature=2)


def ask_question(abstract, persona, number_of_questions=10):
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("{persona}."
                                                        "You are given an abstract of a study to generate questions only."
                                                        "The questions you generate should be simple, singular, and must be directly related to the abstract."
                                                        "Each question must be specific to the particular abstract provided."
                                                        "Ensure no question is a general one that can be answered by multiple abstracts."
                                                        "Create questions in such a way that they will be answered by this abstract only, even if you have 100 different abstracts."
                                                        "The questions should be answerable by just looking at the provided study abstract."
                                                        "For each question, include three fields: 'question', 'part_of_abstract', and 'answer' in Python dictionary format within a Python list. No additional text before or after it."
                                                        "Make sure the output is in 'JSON data format' without printing any type of message."
                                                        "Do not add '\n' to the output or include extra quotation marks; strictly ensure it is a JSON data object only."
                                                      ),
            HumanMessagePromptTemplate.from_template("The study abstract starts here after : \n {text}"),
        ]
    )
    return chat_template.format_messages(text=abstract, persona=persona, number_of_questions=number_of_questions)


def read_abstracts_from_json(file_path,read_no_of_abstract):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[:read_no_of_abstract]#number of study description for which to generate the questions



if __name__ == '__main__':
    personas = ["""You are an undergraduate student at a Primarily Undergraduate Institution taking an Introductory Cloud Computing for Biology. You have biological domain knowledge.""",
                """You are a citizen scientist with a rare disease who has researched on their own. 
                You want to explore data to understand comorbidities and disease outcomes to inform your 
                decisions about treatments and raise disease and treatment awareness in your community.""",
                """
                You are a staff scientist at a research intensive university that needs access to controlled data in order to harmonize phenotypes across studies. 
                You plan to share the harmonization results with a consortium.
                """
                ]
    

def dump_formatted_questions(file_name, data):
    directory='LLM_GeneratedQuestions-1'
    # Ensure the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Full path to the JSON file
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=4)


def question_generation(file_path,list_user_persona,read_no_of_abstract):
    abstracts = read_abstracts_from_json(file_path,read_no_of_abstract)
    for pi in range(len(list_user_persona)):
        persona = list_user_persona[pi]
        llm_questions =[]
        for study_number, study in enumerate(abstracts, start=0):
            print("working on",study_number)
            permalink = study["Permalink"]
            study_name = study_name = permalink.split("/")[-2]
            study_data = {"study_name": study_name,"user_persona": f"persona_{pi}"}
            questions = []
            messages = ask_question(study['Description'], persona, number_of_questions=10)
            result = chat_model.invoke(input=messages)
            #print(result.content)
            questions.append(result.content)
            #print(questions)
            llm_questions.append(study_data)
            study_data["questions"] = questions
        file_name = f"persona_{pi}_questions.json"
        dump_formatted_questions(file_name, llm_questions)


#Path to the study description and number of personas and the no of abstract for which the 
#questions are to be generated
file_path = "/Users/ykale/Documents/Dev/koios/Koios/prompts/99_select_studies.json"
list_user_persona = personas[:3]
read_no_of_abstract = 99
question_generation(file_path,list_user_persona,read_no_of_abstract)