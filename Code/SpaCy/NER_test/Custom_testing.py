import spacy
nlp = spacy.load("en_core_web_lg")

doc = nlp("Donald Trump was President of USA")

doc.ents

from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)

import json

with open('./NER_test/Sample_data/Corona2.json', 'r') as f:
     data = json.load(f)

training_data = []
for example in data['examples']:
    temp_dict = {}
    temp_dict['text'] = example['content']
    temp_dict['entities'] = []
    for annotation in example['annotations']:
        start = annotation['start']
        end = annotation['end']
        label = annotation['tag_name'].upper()
        temp_dict['entities'].append((start, end, label))
    training_data.append(temp_dict)

from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en")
doc_bin = DocBin()

from spacy.util import filter_spans

for training_example in tqdm(training_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("./NER_test/train.spacy")

nlp_ner = spacy.load("model-best")

doc = nlp_ner("While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.")

colors = {"PATHOGEN": "#F67DE3", "MEDICINE": "#7DF6D9", "MEDICALCONDITION": "#a6e22d"}
options = {"colors": colors}

spacy.displacy.render(doc, style="ent", options=options, jupyter=True)