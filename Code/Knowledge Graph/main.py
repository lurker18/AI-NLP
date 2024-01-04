# Load the OpenAI Wikipedia page
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


raw_documents = WikipediaLoader(query = "OpenAI").load()
raw_documents

# Define chunking strategy
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000, chunk_overlap = 20
)

# Chunk the document
documents = text_splitter.split_documents(raw_documents)
for d in documents:
    del d.metadata["summary"]

for doc in documents:
    print(doc.metadata["source"])
    
    
for i in range(len(documents)-1, 0, -1):
    if i == 0:
        continue
    else:
        documents.remove(documents[i])

# Enable Neo4j database
#export OPENAI_API_KEY = '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'
#export NEO4J_USERNAME = 'neo4j'
#export NEO4J_PASSWORD = 'SaA-VGh4DHRCkwA7lqiPumJ8TiiSfqOIazflsjlmEEw'
#export NEO4J_URI = 'neo4j+s://b77e359d.databases.neo4j.io'

# News Articles
# Directory containing your PDF files
directory_path = '/home/lurker18/Desktop/Bio_Relation_Extraction/Dataset/tcnews'

# Initialize PyPDFLoader for each PDF in the directory
loaders = [PyPDFLoader(os.path.join(directory_path, f)) for f in os.listdir(directory_path) if f.endswith('.pdf')]

# Load documents from PDFs
news_docs = []
for loader in loaders:
    news_docs.extend(loader.load())
    
# Prepare the content and metadata for each news article as Document objects
news_articles_data = [
    Document(
        page_content = doc.page_content, # Assuming this is how you access the page content of the document
        metadata = {
            "source" : doc.metadata['source'].removeprefix(directory_path + '/'), # Assuming this is the metadata format
            # Include any other metadata items here
            }
    )
    for doc in news_docs # Assuming news_docs is a list of objects with page_content and metadata
]

all_data = documents + news_articles_data
all_data


# Perform Article Summaries as Relationship Extraction Database
# Initialize the text splitter
rtext_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150)

# Initialize LLM
llm = ChatOpenAI(temperature = 0, model_name = 'gpt-3.5-turbo')

# Define the map prompt template
map_template = """The following is a set of documents
{all_data}
Based on this list of docs, please perform concise summaries while extracting essential relationships for relationships analysis later, please do include dates of actions or events, which are very important for timeline analysis later. Example: "Sam gets fired by the OpenAI board on 11/17/2023 or (Nov. 17th, Friday)", which showcases not only the relationship between Sam and OpenAI, but also when it happens.
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)


# Define the map_chain
map_chain = LLMChain(llm = llm, prompt = map_prompt)

# Extract text from each document
# all_text_data = [doc.page_content for doc in all_data]

# Reduce
reduce_template = """The following is set of summaries:
{all_data}
Take these and distill it into concise summaries of the articles while containing importatnt relationships and events (including the timeline). Example: "Sam gets fired by the OpenAI board on 11/17/2023 or (Nov. 17th, Friday)", which showcases not only the relationship between Sam and OpenAI, but also when it happens.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# ChatPromptTemplate(input_variables = ['all_data'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['all_data'], template='The following is a set of documents:\n{all_data}\nBased on this list of docs, please identify the main themes \nHelpful Answer:'))])

# Run chain
reduce_chain = LLMChain(llm = llm, prompt = reduce_prompt)
combine_documents_chain = StuffDocumentsChain(
    llm_chain = reduce_chain,
    document_variable_name = "all_data" # This should match the variable name in reduce_prompt
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain = combine_documents_chain,
    # If documetns exceed context for 'StuffDocumentsChain'
    collapse_documents_chain = combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max = 4000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain = map_chain,
    # Reduce chain
    reduce_documents_chain = reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name = 'all_data',
    # Return the results of the map steps in the output
    return_intermediate_steps = False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000, chunk_overlap = 0
)

split_docs = text_splitter.split_documents(all_data)

# Run the MapReduce Chain
summarization_results = map_reduce_chain.run(split_docs)
summarization_results

# Store summarization_results to a text file for future use
# Timeline will further be added into the summaries
with open('summary.txt', 'w') as file:
    file.write(str(summarization_results))
    
# Entity and Relationship
import json
import spacy
from collections import Counter
from pathlib import Path
from wasabi import msg
from spacy_llm.util import assemble

# traditional spacy NER (Named Entity Recognition Library)
def split_document_sent(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents] # referencial

# spacy-llm relationship extraction
def process_text(nlp, text, verbose=False):
    doc = nlp(text)
    if verbose:
        msg.text(f"Text: {doc.text}")
        msg.text(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
        msg.text("Relations:")
        for r in doc._.rel:
            msg.text(f"  - {doc.ents[r.dep]} [{r.relation}] {doc.ents[r.dest]}")
    return doc

def run_pipeline(config_path, examples_path=None, verbose=False):
    if not os.getenv("OPENAI_API_KEY"):
        msg.fail("OPENAI_API_KEY env variable was not found. Set it and try again.", exits=1)

    nlp = assemble(config_path, overrides={} if examples_path is None else {"paths.examples": str(examples_path)})

    # Initialize counters and storage
    processed_data = []
    entity_counts = Counter()
    relation_counts = Counter()

    # Load your articles and news data here
    # all_data = news_articles_data + documents

    sents = split_document_sent(summarization_results)
    for sent in sents:
        doc = process_text(nlp, sent, verbose)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        relations = [(doc.ents[r.dep].text, r.relation, doc.ents[r.dest].text) for r in doc._.rel]
        
        # Store processed data
        processed_data.append({'text': doc.text, 'entities': entities, 'relations': relations})

        # Update counters
        entity_counts.update([ent[1] for ent in entities])
        relation_counts.update([rel[1] for rel in relations])

    # Export to JSON
    with open('processed_data.json', 'w') as f:
        json.dump(processed_data, f)

    # Display summary
    msg.text(f"Entity counts: {entity_counts}")
    msg.text(f"Relation counts: {relation_counts}")

# Set your configuration paths and flags
config_path = Path("zeroshot.cfg")
examples_path = None  # or None if not using few-shot
verbose = True

# Run the pipeline
file = run_pipeline(config_path, None, verbose)
        