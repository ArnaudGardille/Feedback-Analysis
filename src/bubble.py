from bubble_api import Field as BubbleField
from bubble_api import BubbleClient
import pandas as pd

try:
    from context import context
except:
    from src.context import context


base_url = "https://vigie.ai"

# bubble_id = os.environ["BUBBLE_ID"]
bubble_id = "1bda18af59187d96f5c1cd4a457db0a3"

BUBBLE_VERSION = "test"

bubble_client = BubbleClient(
    base_url=base_url,
    api_token=bubble_id,
    bubble_version=BUBBLE_VERSION,
)

try:
    COMPANY_ID = bubble_client.get_objects(
        "Company",
        [
            BubbleField("Name") == context["company"],
        ],
    )[0]["_id"]
    print("Retrieved company", context["company"], ":", COMPANY_ID)

except:
    COMPANY_ID = bubble_client.create(
        "Company",
        {
            "Name": context["company"],
        },
    )
    print("Created company", context["company"], ":", COMPANY_ID)

try:
    PROJECT_ID = bubble_client.get_objects(
        "Project",
        [
            BubbleField("Name") == context["project"],
        ],
    )[0]["_id"]
    print("Retrieved project", context["project"], ":", PROJECT_ID)

except:
    PROJECT_ID = bubble_client.create(
        "Project",
        {
            "Name": context["project"],
            "Enjeux": "Permettre de mieux comprendre les attentes des clients, d'identifier les points d'amélioration et de fidéliser la clientèle.",
        },
    )
    print("Created project", context["project"], ":", PROJECT_ID)


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


def get(
    bubble_type, constraints=[BubbleField("company") == COMPANY_ID], max_objects=None
):
    df = pd.DataFrame(
        bubble_client.get_objects(
            bubble_type,
            constraints,
            max_objects=max_objects,
        )
    )
    df.drop(["Modified Date", "Created Date", "Created By"], axis=1, inplace=True)
    #df["Modified Date"] = pd.to_datetime(df["Modified Date"])
    #df["Created Date"] = pd.to_datetime(df["Created Date"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df

def create_or_update(bubble_type, d, field="Name"):
    data = bubble_client.get(bubble_type, constraints= [BubbleField(field) == d[field]])
    if len(data)>0:
        bubble_id = data[0]["_id"]
        bubble_client.update_object("Company", bubble_id, d)
    else:
        bubble_id = bubble_client.create("Company", d)