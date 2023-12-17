# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Universal-NER/UniNER-7B-all")
model = AutoModelForCausalLM.from_pretrained("Universal-NER/UniNER-7B-all")
