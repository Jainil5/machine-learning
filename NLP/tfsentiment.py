import json

file =  open("NLP/sarcasm.json") 
datastore = json.load(file)


sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

