from django.shortcuts import render
import pandas as pd
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModel
from django.http import JsonResponse

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

df = pd.read_csv('/workspaces/codespaces-django/hello_world/core/prop_data.csv')

property_dataset = Dataset.from_pandas(df)

def concatenate_text(data):
    
    return {"text": data['amt_name'] + '\n' + data['description_x'] + '\n' + data['location'] + '\n' + data['address'] + '\n' + data['description_y'] + '\n' + 
           
           data['title'] + '\n' + data['link'] + '\n' + data['main_video'] + '\n' + data['display_flag']}


property_dataset = property_dataset.map(concatenate_text)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embeddings_dataset = property_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)

embeddings_dataset.add_faiss_index(column='embeddings')

def index(request, question):

    question_embedding = get_embeddings([question]).numpy()

    scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    result = samples_df['text']

    return JsonResponse(
        result.to_dict()    
    )

