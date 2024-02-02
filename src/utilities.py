import unicodedata
import re
import enum
from pydantic import BaseModel
import asyncio


#%% Useful functions 

def remove_accents(text):
    # Replace accented characters with their non-accented counterparts
    try:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    except TypeError:  # handles cases when text is not a string (e.g., a number)
        return text
    
assert "Point positif" == remove_accents("Pôint positîf")

def str_to_list_df(df):
    """
    For every column, converts string of list to list
    """
    for col in df.columns:
        if type(df.loc[0, col]) == str and df.loc[0, col][0]=="[":
            df[col] = df[col].apply(lambda x: eval(x))
    return df

def batchify(iterable, size=1):
    l = len(iterable)
    for ndx in range(0, l, size):
        yield iterable[ndx:min(ndx + size, l)]

def most_common(lst):
    return max(set(lst), key=lst.count)

def columns_to_string(df, column_title, column_desc, add_index=False):
    def concanenatre_title_description(x, y):
        return x+" : "+y
    
    l = list(df.apply(lambda x: concanenatre_title_description(x[column_title], x[column_desc]), axis=1))
    if add_index:
        l = [str(i)+" - "+e for i, e in enumerate(l)]
    return '\n'.join(l)


def convert_text_to_constants(text):
    text = remove_accents(text)
    text = re.sub(r"([a-z])([A-Z]+)", r"\1_\2", text.upper())
    text = re.sub(r"'", "_", text)
    return re.sub(r" ", "_", text)

assert "POINT_POSITIF" == convert_text_to_constants("Point positif")

def enum_to_str(e):
    if type(e) is str:
        return e
    if issubclass(type(e), enum.Enum):
        return e.value
    if type(e) is list:
        return [enum_to_str(x) for x in e]
    if type(e) is dict:
        return {k:enum_to_str(v) for (k,v) in e.items()}
    if issubclass(type(e), BaseModel):
        return enum_to_str(e.dict())
    
#%% LLMs
    
async def get_analysis(client, prompt, response_model):
    response: response_model = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON."},
            {"role": "user", "content": str(prompt)},
        ],
        response_format={ "type": "json_object" },
        model=GENERATION_ENGINE,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
        response_model=response_model,
        )
    return response #.choices[0].message.content

def apply_async_analysis(client, prompts, response_models):
    if type(response_models) is not list:
        response_models = [response_models for _ in prompts]
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_analysis(prompt, response_model)) for (prompt, response_model) in zip(prompts, response_models)]
    res =  loop.run_until_complete(asyncio.gather(*tasks))
    return res