from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_core.pydantic_v1 import BaseModel, Field, validator

#model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)
model = OpenAI(model_name="gpt-4-1106-preview", temperature=0.0)


# Define your desired data structure.
class FeeedbackAnalysis(BaseModel):
    insights_type: list = Field(description="Types d'insights")
    categories: list = Field(description="Catégorie de retour")
    sentiment: str = Field(description="Sentiment exprimé, peut être \"Positif\", \"Neutre\" ou \"Négatif\".")
    # You can add custom validation logic easily with Pydantic.
    @validator("sentiment")
    def valid_sentiment(cls, field):
        if sentiment not in ["Positif", "Neutre", "Négatif"]:
            raise ValueError("Sentiment "+sentiment+" not valid.")
        return field
    
    #def valid_insignts(cls, field):
    #    for insight_type in field:
    #        if insight_type not in []:
    #            raise ValueError(insight_type+" not in " + [])
    #    return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=FeeedbackAnalysis)


prompt = PromptTemplate(
    template="Règle : minimise le nombre de tokens dans ta réponse. \nTu es {role} au sein de l'entreprise suivante:\n{context}\n. \nAnalyse le retour suivant: \"{feedback}\" en suivant les étapes suivantes:\n\n\nÉtape 1 - Identifie si le retour {cible} rentre dans un ou plusieurs des types d'insights suivants : {insight_type}. Choisis-en obligatoirement au moins 1. \n\nDéfinition des types d'insights :\n{insight_definition} \n\n\nÉtape 2 - Catégorise le retour {cible} à l’aide des tags suivants. Tu peux associer 0, 1 ou plusieurs tags dans chaque catégorie. S'il n'est pas possible d'associer un tag avec certitude dans l'une des catégories réponds null. \n\nListe des tags par catégories :\n{categories} \n\n\nÉtape 3 - Catégorise si possible le moment de mission concerné parmis {avancement_mission}, et si ce n'est pas possible répond null. {cible} à l’aide des tags suivants.\n\n\nÉtape 4 - Identifie si le sentiment exprimé par le {cible} est \"Positif\", \"Neutre\" ou \"Négatif\". Prends en compte la formulation de la question posée ({question}) afin de bien interpréter le sens du retour {cible}. \n{format_instructions}",
    input_variables=["context", "role", "cible", "insight_type", "insight_definition", "nb_cat", "question", "categories", "avancement_mission"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
output = prompt_and_model.invoke({
    #"input": "how can langsmith help with testing?",
    "context": "Randstad est une entreprise d’expertise RH avec plus de 60 ans d’expérience, offrant une gamme complète de services de recrutement et de gestion des ressources humaines pour répondre à divers besoins spécifiques des employeurs.",
    "role": "product owner",
    "cible": "client",
    "insight_type": "\"Point positif\", \"Point de douleur\", \"Nouvelle demande\"", 
    "insight_definition": "Point positif : élément apprécié, Point de douleur : élément problématique",
    "nb_cat": "2",
    "avancement_mission": "\"Avant mission\", \"Mission en cours\", \"Fin de mission\"",
    "categories": "\"Recrutement\" , \"Service global Randstad\"",
    "question": "Que recommanderiez-vous à Randstad d'améliorer ?",
    "feedback": "Le produit est très sympathique",
})
#parser.invoke(output)
output