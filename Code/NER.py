import spacy
import medspacy
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent

nlp = medspacy.load()
print(nlp.pipe_names)

text = """
Past Medical History:
1. Atrial fibrillation
2. Type II Diabetes Mellitus

Assessment and Plan:
There is no evidence of pneumonia. Continue warfarin for Afib. Follow up for management of type 2 DM.
"""

target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_rules = [
    TargetRule("atrial fibrillation", "PROBLEM"),
    TargetRule("atrial fibrillation", "PROBLEM", pattern=[{"LOWER": "afib"}]),
    TargetRule("pneumonia", "PROBLEM"),
    TargetRule("Type II Diabetes Mellitus", "PROBLEM", 
              pattern=[
                  {"LOWER": "type"},
                  {"LOWER": {"IN": ["2", "ii", "two"]}},
                  {"LOWER": {"IN": ["dm", "diabetes"]}},
                  {"LOWER": "mellitus", "OP": "?"}
              ]),
    TargetRule("warfarin", "MEDICATION")
]
target_matcher.add(target_rules)

doc = nlp(text)
visualize_ent(doc)