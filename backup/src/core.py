import json
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# llm = Ollama(model="llama3")
chat_model = ChatOllama(model="llama3",base_url="http://localhost:60423")


def ask_question(abstract, persona, number_of_questions=10):
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("{persona}."
                                                      "You are given the abstract of a study to generate questions."
                                                      "The questions you generate should be simple and singular."
                                                      "The questions should be answerable by just looking at the study abstract provided. "
                                                      "For each question you generate, include the part of the abstract that answers it right below and also the answer to the question."
                                                      "Generate all possible questions that will cover the complete abstract. Output your response in json list."
                                                      "Do not output additional text other than the json list."
                                                      ),
            HumanMessagePromptTemplate.from_template("Abstract: \n {text}"),
        ]
    )
    return chat_template.format_messages(text=abstract, persona=persona, number_of_questions=number_of_questions)
#"Generate {number_of_questions} possible questions. Output your response in json list."
def read_abstracts_from_json(file_path):
    #print(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[:2]


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

    file_path = "/Users/ykale/Documents/Dev/koios/Koios/prompts/study_description.json"
    abstracts = read_abstracts_from_json(file_path)

    questions = []

    for study in abstracts:
        messages = ask_question(study['Description'], personas[0], number_of_questions=10)
        result = chat_model.invoke(input=messages)
        questions.append(result)
    print(questions)

    # Write questions to a JSON file
    with open('questions.json', 'w') as file:
        json.dump(questions, file)
    #print("/n/n",messages)
    #print(messages.__str__())
    #result = chat_model.invoke(input=messages)
    #result.pretty_print()
