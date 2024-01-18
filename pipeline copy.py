import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
from sentence_transformers import SentenceTransformer
import pprint
from typing import List

from langchain_community.llms import Ollama
from langchain.output_parsers.regex_dict import RegexDictParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
import asyncio

import itertools
from copy import copy
from tqdm.notebook import tqdm, trange
import streamlit as st
from stqdm import stqdm
from sklearn.cluster import KMeans
import openai
openai.api_key = "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"


#OPENAI_API_KEY = "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"
model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, api_key="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")
client = AsyncOpenAI()

import nest_asyncio
nest_asyncio.apply()

embedding_model = SentenceTransformer('OrdalieTech/Solon-embeddings-large-0.1')
GENERATION_ENGINE = "gpt-4-1106-preview"
EMBEDDING_ENGINE = "text-embedding-ada-002"

async def async_generate(chain, params):
    return await chain.arun(params)

#%% Extracting insights and categories from feedbacks

class FirstInsight(BaseModel):
    insight_types: List[str] = Field(description="Types de l'insight")
    content: str = Field(description="Point intéressant a retenir du commentaire.")

    def __str__(self):
        return '- ' + self.content + "\nTypes: " + ', '.join(self.insight_types)

class Feedback(BaseModel):
    insights_list: List[FirstInsight] = Field(description="Contenu et type des insights")
    sentiment: str = Field(description="Sentiment exprimé, peut être 'Positif', 'Neutre' ou 'Négatif'.")
    content = ""
    # You can add custom validation logic easily with Pydantic.
    #@validator("sentiment")
    #def valid_sentiment(cls, field):
    #    if field not in ["Positif", "Neutre", "Négatif"]:
    #        raise ValueError("Sentiment "+field+" not valid.")
    #    return field
    
    def __str__(self):
        return "Commentaire: \""+ self.content+"\"\n\nSentiment: "+self.sentiment+"\n\nInsights: \n"+"\n\n".join([str(i) for i in self.insights_list])
    


prompt_template_feedback_initial = """Tu es {role} au sein de l'entreprise suivante: 
{context}
Pour le retour {cible}, effectue les étapes suivantes: 

Étape 1 - Identifie si il rentre dans un ou plusieurs des catégories d'insights suivantes : {insight_type}, dont la definition est: 
{insight_definition} 

Étape 2 - Catégorise les si possible avec les tags suivants: {categories} 

Étape 3 - Catégorise si possible le moment de mission concerné parmis {avancement_mission}.

Étape 4 - Identifie si le sentiment exprimé par le {cible} est \"Positif\", \"Neutre\" ou \"Négatif\". Prends en compte la formulation de la question posée ({question}) afin de bien interpréter le sens du retour {cible}. 

Étape 5 - Identifie le ou les éventuelles insights que tu aurais envie de faire remonter à ton équipe. Ils doivent être des phrase grammaticalement correcte, et faire correspondre intelligement le commentaire au context de l'entreprise. Si rien d'intéressant ne peut être conclu, laisse la liste vide. Si plusieurs points distinguables sont a relever, formule plusieurs insights. Ces insights sont voués a être commun a d'autres commentaires qui seront analysés.
Par exemple, pour le commentaire suivant:
'''
{exemple_commentaire}
'''
on voudrait faire remonter les points suivants:
'''
{exemple_insights}
'''
Ces insights sont en effet distincts, pertinent par rapport au commentaire et au context de l'entreprise, et important à prendre en compte.

Voici le commentaire que tu dois traiter: {feedback}

{format_instructions}  
"""

#def create_feedback_categoriser(invocation):
@st.cache_data
def feedback_categoriser(invocation, feedback): 
    invocation = copy(invocation)
    invocation['feedback'] = feedback
    
    output = prompt_and_model_feedback.invoke(invocation)
    output.content = feedback
    return output
    
    #return lambda feedback:feedback_categoriser(invocation, feedback)


#%% Merging insights
class DeducedInsight(BaseModel):
    childens: List[int] = Field(description="Index des insights mineurs qui ont été résumés en cet insight.")
    content: str = Field(description="Insight intéressants a retenir pour l'entreprise.")

    def __str__(self):
        return '- ' + self.content + '\n Enfants:' + str(self.childens)


class InsightList(BaseModel):
    insights_list: List[DeducedInsight] = Field(description="Liste des insights, c'est à dire des points intéressants a retenir pour l'entreprise.")
    # You can add custom validation logic easily with Pydantic.
    
    def __str__(self):
        return "Insights: \n"+"\n\n".join([str(i) for i in self.insights_list])
    

prompt_template_insight_initial = """Tu es {role} au sein de l'entreprise suivante: 
{context}

Une liste d'insights mineurs a été identifiée à partir de retours clients. 
Résume les en des insights majeurs qui te semblent important à faire remonter au sein de l'entreprise. Ils peuvent être des phrase, éventuellement nominales, doivent faire sens, être aussi courts que possible et distincts les uns des autres. 
Ensuite, associe à chaque insight majeur l'indice des insights mineurs qui lui sont associés. Un insights mineur peut être associé à plusieurs insights majeurs. Vérifie bien que les indices correspondent. 
L'ordre des insights mineurs est aléatoire, et ne doit pas avoir d'importance dans ta réponse. 

Voici les insights mineurs que tu dois regrouper: 

{insights}

{format_instructions}  """

#def create_insights_merger(invocation):
@st.cache_data
def insights_merger(invocation, cluster): 
    invocation = copy(invocation)
    invocation['insights'] = '\n'.join([str(i)+": "+s for i, s in enumerate(cluster["content"])])
    
    output = prompt_and_model_insight.invoke(invocation)
    
    #pd.concat([  for insight in output.insights_list])
    dfs = pd.DataFrame({
        "related_feedbacks":[list(itertools.chain.from_iterable(cluster.iloc[insight.childens]['related_feedbacks'])) for insight in output.insights_list],
        "content":[insight.content for insight in output.insights_list],
        "children":[list(cluster.iloc[insight.childens].index) for insight in output.insights_list],
        })
    
    return dfs #, reduction
    
    #return lambda feedback:insights_merger(invocation, feedback)



#%% Async

async def get_embedding(text):
    response = await client.embeddings.create(input=text, model=EMBEDDING_ENGINE)
    return response.data[0].embedding

def apply_async_get_embedding(dfi):
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_embedding(row['Comment'])) for _, row in dfi.iterrows()]
    return loop.run_until_complete(asyncio.gather(*tasks))

async def get_analysis(prompt):
    response = await client.completions.create(
        prompt=prompt
        , model=GENERATION_ENGINE)
    return response.choices[0].text

def apply_async_analysis(prompts):
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_analysis(prompt)) for prompt in prompts]
    return loop.run_until_complete(asyncio.gather(*tasks))
    
#%%

#warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

columns_feedback = ["add_ons", "date", "content", "project", "sentiment", "source", "tag_1", "tag_2", "tag_3", "tag_4", "tag_5"]
columns_insights = ["related_feedbacks", "content", "childrens","project", "source"]

uploaded_file = st.file_uploader("CSV containing the feedbacks", type="csv")
if uploaded_file is not None:
    initial_feedbacks_df = pd.read_csv(uploaded_file)
    #if 'sentiment' not in feedbacks_df.columns:
    st.dataframe(initial_feedbacks_df, use_container_width=True)
    feedbacks_column = st.selectbox("which columns contains the feedbacks?", initial_feedbacks_df.columns)
    feedbacks_df = copy(initial_feedbacks_df)[:3]
    feedbacks_df['sentiment'] = ""
    #project = st.text_input("Project:")
    #source = st.text_input("Source:")
    prompt_template_feedback = st.text_area("Prompt", prompt_template_feedback_initial)

    context = st.text_area("Context de l'entreprise", "Metro AG, ou Metro Group, est un groupe de distribution allemand. Il est notamment connu pour ses enseignes de vente en gros, cash & carry, aux professionnels dans de nombreux pays (Metro Cash & Carry et Makro).")
    role = st.text_input("Role", "product owner")
    cible = st.text_input("Cible", "client")
    question = st.text_input("Question", "Que recommanderiez-vous à Metro d'améliorer ?")

    exemple_commentaire = st.text_area("Exemple de commentaire:", "je suis exclusif metro je n ai aucun representant j achetais jusqu a present tout metro par facilite mais je suis tres souvent décue par la reponse ha non on n en a pas cela arrive demain je pense que depuis le covid tout le monde ou presque s en fou!!!")

    initial_examples_insights_df = pd.DataFrame([
        {"Insights qui devraient en découler": "Déceptions face aux retard de livraison"},
        {"Insights qui devraient en découler": "Impression d'une baisse de qualité du service depuis le Covid"},
    ])
    examples_insights_df = st.data_editor(initial_examples_insights_df, num_rows="dynamic",  use_container_width=True)
    
    if st.button("Run"):

        initial_examples_insights_df = pd.DataFrame([
            {"Insights qui devraient en découler": "Déceptions face aux retard de livraison"},
            {"Insights qui devraient en découler": "Impression d'une baisse de qualité du service depuis le Covid"},
        ])
        #examples_insights_df = st.data_editor(initial_examples_insights_df, num_rows="dynamic",  use_container_width=True)
    
        prompt_template_insight = st.text_area("Prompt", prompt_template_insight_initial)

        feedback_context = {
            "context": context,
            "role": role,
            "cible": cible,
            "insight_type": "\"Point positif\", \"Point de douleur\", \"Nouvelle demande\"", 
            "insight_definition": "Point positif : élément apprécié, Point de douleur : élément problématique",
            "nb_cat": "2",
            "avancement_mission": "\"Avant mission\", \"Mission en cours\", \"Fin de mission\"",
            "categories": "\"Recrutement\" , \"Service global\"",
            "question": question,
            "exemple_commentaire": exemple_commentaire,
            "exemple_insights": list(examples_insights_df)
        }
        
        feedback_parser = PydanticOutputParser(pydantic_object=Feedback)

        prompt_feedback = PromptTemplate.from_template(
            template= prompt_template_feedback,
            #template= "Règle : minimise le nombre de tokens dans ta réponse.  \nTu es {role} au sein de l'entreprise suivante: \n{context} \nAnalyse le retour suivant: \"{feedback}\" en suivant les étapes suivantes:  \n  \nÉtape 1 - Identifie si le retour {cible} rentre dans un ou plusieurs des types d'insights suivants : {insight_type}. Choisis-en obligatoirement au moins 1. Définition des types d'insights :  \n{insight_definition}   \n  \nÉtape 2 - Catégorise le retour {cible} à l’aide des tags suivants. Tu peux associer 0, 1 ou plusieurs tags dans chaque catégorie. Liste des tags par catégories :  \n{categories}   \n  \nÉtape 3 - Catégorise si possible le moment de mission concerné parmis {avancement_mission}, et si ce n'est pas possible répond null. {cible} à l’aide des tags suivants.  \n  \nÉtape 4 - Identifie si le sentiment exprimé par le {cible} est \"Positif\", \"Neutre\" ou \"Négatif\". Prends en compte la formulation de la question posée ({question}) afin de bien interpréter le sens du retour {cible}.   \n",
            #input_variables= ["context", "role", "cible", "insight_type", "insight_definition", "nb_cat", "avancement_mission", "categories", "question", "feedback"]
            partial_variables= {"format_instructions": feedback_parser.get_format_instructions()},
        )

        prompts = []
        for feedback in feedbacks_df[feedbacks_column]:
            prompt = copy(feedback_context)
            prompt["feedback"] = feedback
            prompts.append(prompt)
        responses = apply_async_analysis(prompts)
        responses = [feedback_parser(rep) for rep in responses]
        feedbacks_df["Sentiment"] = [rep.sentiment for rep in responses]
        feedbacks_df["Insights"] = []

        k=0
        insights = []
        for i, rep in enumerate(responses):
            for j, insight in enumerate(rep.insights_list):
                insights.append(insight)
                feedbacks_df["Insights"].iloc[i].append(str(k))
                k += 1


        #prompt_and_model_feedback = prompt_feedback | model | feedback_parser

        #feedback_categoriser = create_feedback_categoriser(feedback_context)
        
        

        #st.session_state.feedbacks_df = feedbacks_df
        st.dataframe(feedbacks_df, use_container_width=True)

        insights_df = pd.DataFrame({"Content":insights})
        st.dataframe(insights_df, use_container_width=True)

        insights_context = {
            "context": context,
            "role": role,
        }
        #insights_merger = create_insights_merger(insights_context)

        # Merging stuff

        minimisation_steps = st.number_input("Minimisation steps", 5)
        cluster_desired_size = st.number_input("Cluster desired size", 30)
        nb_insight_stop = st.number_input("Stop if the number of insights is below", 10)

        if st.button("Launch clustering"):

            parser_insight = PydanticOutputParser(pydantic_object=InsightList)

            prompt_insight = PromptTemplate.from_template(
                template= prompt_template_insight,
                #template= "Règle : minimise le nombre de tokens dans ta réponse.  \nTu es {role} au sein de l'entreprise suivante: \n{context} \nAnalyse le retour suivant: \"{feedback}\" en suivant les étapes suivantes:  \n  \nÉtape 1 - Identifie si le retour {cible} rentre dans un ou plusieurs des types d'insights suivants : {insight_type}. Choisis-en obligatoirement au moins 1. Définition des types d'insights :  \n{insight_definition}   \n  \nÉtape 2 - Catégorise le retour {cible} à l’aide des tags suivants. Tu peux associer 0, 1 ou plusieurs tags dans chaque catégorie. Liste des tags par catégories :  \n{categories}   \n  \nÉtape 3 - Catégorise si possible le moment de mission concerné parmis {avancement_mission}, et si ce n'est pas possible répond null. {cible} à l’aide des tags suivants.  \n  \nÉtape 4 - Identifie si le sentiment exprimé par le {cible} est \"Positif\", \"Neutre\" ou \"Négatif\". Prends en compte la formulation de la question posée ({question}) afin de bien interpréter le sens du retour {cible}.   \n",
                #input_variables= ["context", "role", "cible", "insight_type", "insight_definition", "nb_cat", "avancement_mission", "categories", "question", "feedback"]
                partial_variables= {"format_instructions": parser_insight.get_format_instructions()},
            )

            prompt_and_model_insight = prompt_insight | model | parser_insight

            insights = copy(insights_df)
            insight_layers = [copy(insights_df)]

            for step in range(minimisation_steps):

                num_clusters = 1 + len(insights) // cluster_desired_size

                st.markdown("Step"+ str(step)+ ": processing"+ str(num_clusters) + "clusters")
                if len(insights) <= nb_insight_stop:
                    st.markdown("Everything is merged into a single cluster")
                    break

                sentence_embeddings = embedding_model.encode(insights['content'])

                clustering_model = KMeans(n_clusters=num_clusters)
                clustering_model.fit(sentence_embeddings)
                cluster_assignment = clustering_model.labels_
                insights["cluster"] = cluster_assignment
                
                st.markdown("Cluster sizes:" + str(list(insights.groupby(['cluster']).count()["content"])))

                new_insights = []
                for cluster_id in stqdm(range(max(cluster_assignment)+1)):
                    cluster = insights[insights['cluster'] == cluster_id]
                    new_insights.append(insights_merger(cluster))


                new_insights = pd.concat(new_insights)
                new_insights.reset_index(drop=True, inplace=True)
                reduction = len(new_insights)/len(insights)
                st.markdown("Number of new insights:"+ str(len(new_insights)))
                st.markdown("Reduction in the number of insights:" + "%d" % int((len(new_insights)/len(insights))*100) + "%")
                insight_layers.append(copy(new_insights))
                insights = new_insights

                st.dataframe(insights, use_container_width=True)

                




