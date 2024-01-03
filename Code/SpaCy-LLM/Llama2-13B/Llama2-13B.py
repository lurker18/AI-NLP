from spacy_llm.util import assemble

# Assemble a spaCy pipeline from the config
nlp = assemble("Code/SpaCy-LLM/Llama2-13B/config.cfg")

# Use this pipeline as you would normally
doc = nlp("Addison's disease (also known as primary adrenal insufficiency or hypoadrenalism) is a rare disorder of the adrenal glands. The adrenal glands are two small glands that sit on top of the kidneys. They produce essential hormones: cortisol, aldosterone and adrenaline. In Addison's disease, the adrenal gland is damaged, and not enough cortisol and aldosterone are produced. About 8,400 people in the UK have Addison's disease. It can affect people of any age, although it's most common between the ages of 30 and 50.")
print(doc)
for ent in doc.ents:
    print(ent, ent.label_)
