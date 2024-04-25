import json
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# llm = Ollama(model="llama3")
chat_model = ChatOllama(model="llama3",base_url="http://localhost:11434")


def ask_question(abstract, persona, number_of_questions=10):
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("{persona}."
                                                      "You are given the abstract of a study to generate questions."
                                                      "The questions you generate should be simple and singular."
                                                      "The questions should be answerable by just looking at the study abstract provided. "
                                                      "For each question you generate, include the part of the abstract that answers it right below and also the answer to the question."
                                                      "Generate {number_of_questions} possible questions. Output your response in json list."
                                                      "Do not output additional text other than the json list."
                                                      ),
            HumanMessagePromptTemplate.from_template("Abstract: \n {text}"),
        ]
    )
    return chat_template.format_messages(text=abstract, persona=persona, number_of_questions=number_of_questions)




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

    abstract = ("<p> The Age-Related Eye Disease Study (AREDS) was initially designed as a long-term multi-center, "
                "prospective study of the clinical course of age-related macular degeneration (AMD) and age-related cataract. "
                "In addition to collecting natural history data, AREDS included a clinical trial of high-dose vitamin "
                "and mineral supplements for AMD and a clinical trial of high-dose vitamin supplements for cataract. "
                "AREDS participants were 55 to 80 years of age at enrollment and had to be free of any illness or "
                "condition that would make long-term follow-up or compliance with study medications unlikely or difficult. "
                "On the basis of fundus photographs graded by a central reading center, best-corrected visual acuity"
                " and ophthalmologic evaluations, 4,757 participants were enrolled in one of several AMD categories,"
                " including persons with no AMD. </p> <p> The clinical trials for AMD and cataract were conducted "
                "concurrently. AREDS participants were followed on the clinical trials for a median time of 6.5 years."
                " Subsequent to the conclusion of the clinical trials, participants were followed for an additional 5 years and "
                "natural history data were collected. The AREDS research design is detailed in AREDS Report 1. "
                "AREDS Report 8 contains the mainline results from the AMD trial; AREDS Report 9 contains the "
                "results of the cataract trial. </p> <p> Blood samples were also collected from 3,700+ AREDS "
                "participants for genetic research. Genetic samples from 600 AREDS participants (200 controls, 200 "
                "Neovascular AMD cases, and 200 Geographic Atrophy cases) were selected using data available in March "
                "2005 and then were evaluated with a genome-wide scan. These data, as well as selected phenotypic data, "
                "were made available in the dbGaP. DNA from AREDS participants, which is currently being stored in the "
                "AREDS Genetic Repository, is available for research purposes. However, not all of the 3,700+ AREDS "
                "participants who submitted a blood sample currently have DNA available. </p> <p> "
                "In addition to including the data from the genome-wide scan on the 600 original samples, this second "
                "version of the AREDS dbGaP database provides a comprehensive set of data tables with extensive clinical "
                "information collected for the 4,757 participants who participated in AREDS. The tables include information"
                " collected at enrollment/baseline, during study follow-up, fundus and lens pathology, nutritional "
                "estimates, quality of life measures and measures of morbidity and mortality. </p> <p>In <b>November "
                "2010</b>, over 72,000 high quality fundus and lens photographs of 595 AREDS participants "
                "(of the original 600 selected for the genome-wide scan) were made available in the AREDS dbGaP. "
                "In addition to the genome-wide scan data, the fundus and lens grading data for these participants are "
                "also available through the AREDS dbGaP. Details about the ocular photographs that are available may be "
                "found in the document \"Age-Related Eye Disease Study (AREDS) <a href=\"GetPdf.cgi?id=phd003307.1\" "
                "target=\"_blank\">Ocular Photographs</a>\". </p> <p> In <b>January 2012</b>, a measure of daily sunlight "
                "exposure was added in a separate &#34;sunlight&#34; table. Furthermore, the &#34;followup&#34; table has been revised. The visual acuity for the right eye was inadvertently missing at odd-numbered visits (01, 03, 05, etc.). This data is now part of the table. </p> <p>In <b>February 2014</b> over 134,500 high-quality fundus photographs (macular field F2) of 4613 AREDS participants were added to the existing AREDS dbGaP resource. The AREDS dbGaP image archive already contains over 72,000 high quality fundus and lens photographs of 595 AREDS participants for whom dbGaP-accessible genotype data exist. Information about the available ocular photographs found in the document \"Age-Related Eye Disease Study (AREDS) <a href=\"GetPdf.cgi?id=phd003307.1\" target=\"_blank\">Ocular Photographs</a>\" has been updated with an addendum. </p> <p> It is hoped that this resource will better help researchers understand two important diseases that affect an aging population. These data may be applied to examination and inference on genetic and genetic-environmental bases for age-related diseases of public health significance and may also help elucidate the clinical course of both conditions, generate hypotheses, and aid in the design of clinical trials of preventive interventions. </p><p> <b>Definitions of Final AMD Phenotype Categories</b><br/> Please see <a href=\"GetPdf.cgi?id=phd001138\" target=\"_blank\">phd001138.1</a> for a detailed description of how AREDS participants&#39; final AMD phenotype was categorized. </p> <p> <b>User&#39;s Guide for AREDS Phenotype Data</b><br/> A detailed User&#39;s Guide for the AREDS phenotype data is available. This <a href=\"GetPdf.cgi?id=phd001552.1\" target=\"_blank\"><b>User&#39;s Guide</b></a> is meant to be a comprehensive document which explains the complexities of the AREDS data. <i>It is recommended that all researchers using AREDS phenotype data make use of this User&#39;s Guide</i>. </p>")

    messages = ask_question(abstract, personas[2], number_of_questions=10)
    # print(messages.__str__())
    result = chat_model.invoke(input=messages)
    result.pretty_print()