from bubble_api import Field as BubbleField
from bubble_api import BubbleClient
import pandas as pd
import os

base_url = "https://feedback-analysis.bubbleapps.io" #https://blumana.app" 

#bubble_id = os.environ["BUBBLE_ID"]
bubble_id = "1bda18af59187d96f5c1cd4a457db0a3"

BUBBLE_VERSION = 'test'

bubble_client = BubbleClient(
    base_url=base_url,
    api_token=bubble_id,
    bubble_version=BUBBLE_VERSION, 
)

def deduce_backend_type(insight_type):
    if insight_type == "1698433300252x835626794232717300":
        return "pain"
    elif insight_type == "1698433290120x936044292663509300":
        return "positive"   
    elif insight_type == "1698433314230x619003097145126100":
        return "feature"  
    elif insight_type == "1698433323222x402426615286320700":
        return "bug"   
    print("Incorrect type:", insight_type)

def get(bubble_type, source_id=None, company_id=None):
    df = pd.DataFrame(bubble_client.get_objects(
        bubble_type,
        [
            BubbleField("source") == source_id,
            BubbleField("company") == company_id,
            ],
    ))
    df['Modified Date'] = pd.to_datetime(df['Modified Date'])
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    return df
