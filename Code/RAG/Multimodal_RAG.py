import wikipedia
from tqdm import tqdm
import shutil
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils import embedding_functions
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer

images_path = '/home/lurker18/Documents/flowers'
images_classes = os.listdir(images_path)

new_path = '/home/lurker18/Desktop/RAG_Bio_Acronym/Dataset/flowers'
if not os.path.exists(new_path):
    os.mkdir(new_path)

for cls in tqdm(images_classes):
    cls_path = os.path.join(images_path, cls)
    new_cls_path = os.path.join(new_path, cls)
    if not os.path.exists(new_cls_path):
        os.mkdir(new_cls_path)
    for image in os.listdir(cls_path)[:10]:
        image_path = os.path.join(cls_path, image)
        new_image_path = os.path.join(new_cls_path, image)
        shutil.copy(image_path, new_image_path)

images_classes

wiki_titles = { # the key is images class and the value is wiki title
    'daisy' : 'Bellis perennis',
    'dandelion' : 'Taraxacum',
    'lotus' : 'Nelumbo nucifera',
    'rose' : 'Rose',
    'sunflower' : 'Common sunflower',
    'tulip' : 'Tulip',
    'bellflower' : 'Campanula'
}

# each class has 10 images and one text file content from the wiki page
for cls in tqdm(images_classes):
    cls_path = os.path.join(new_path, cls)
    # page_content = wikipedia.page(wiki_titles[cls], auto_suggest = False).content
    page_content = wikipedia.summary(wiki_titles[cls], auto_suggest = False)

    if not os.path.exists(cls_path):
        print("Creating {} folder".format(cls))
    else:
        # save the text file
        files_name = cls + '.txt'
        with open(os.path.join(cls_path, files_name), 'w') as f:
            f.write(page_content)

# Defining the Vector DB
client = chromadb.PersistentClient(path = 'DB')
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection_images = client.get_collection(
    name = 'multimodal_collection_images',
    embedding_function = embedding_function,
    data_loader = image_loader
)

collection_text = client.get_collection(
    name = 'multimodal_collection_text',
    embedding_function = embedding_function,
)

# Get the uris to the images
IMAGE_FOLDER = '/home/lurker18/Desktop/RAG_Bio_Acronym/Dataset/flowers/all_data'
image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
ids = [str(i) for i in range(len(image_uris))]

collection_images.add(ids = ids, uris = image_uris)

# Test some flower samples
retrieved_tulips = collection_images.query(query_texts = ['tulip'], include = ['data'], n_results = 3)
for image in retrieved_tulips['data'][0]:
    plt.imshow(image)
    plt.axis('off')
    plt.show()

retrieved_bellflower = collection_images.query(query_texts = ['bellflower'], include = ['data'], n_results = 3)
for image in retrieved_bellflower['data'][0]:
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Test multimodality data of img-to-img query - Daisy part
query_image = np.array(Image.open('/home/lurker18/Documents/flowers/daisy/0.jpg'))
print("Query Image")
plt.imshow(query_image)
plt.axis('off')
plt.show()

print("Results")
retrieved_daisy = collection_images.query(query_images = [query_image], include = ['data'], n_results = 3)
for image in retrieved_daisy['data'][0][1:]:
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Test multimodality data of img-to-img query - Rose part
query_image = np.array(Image.open('/home/lurker18/Documents/flowers/rose/0444a369fb.jpg'))
print("Query Image")
plt.imshow(query_image)
plt.axis('off')
plt.show()

print("Results")
retrieved_rose = collection_images.query(query_images = [query_image], include = ['data'], n_results = 3)
for image in retrieved_rose['data'][0][1:]:
    plt.imshow(image)
    plt.axis('off')
    plt.show()

default_ef = embedding_functions.DefaultEmbeddingFunction()
text_path = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if image_name.endswith('.txt')])
text_path

list_of_text = []
for text in text_path:
    with open(text, 'r') as f:
        text = f.read()
        list_of_text.append(text)

ids_txt_list = ['id' + str(i) for i in range(len(list_of_text))]
ids_txt_list

collection_text.add(documents = list_of_text, ids = ids_txt_list)

results = collection_text.query(
    query_texts = ["What is the bellflower?"],
    n_results = 1
)
results
collection_text.count()

model_id = '/media/lurker18/HardDrive/HuggingFace/models/llava-hf/llava-1.5-7b-hf'
model = LlavaForConditionalGeneration.from_pretrained(model_id)
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

image_file = new_path + '/daisy/918d49898e.jpg'
raw_image = Image.open(image_file)
plt.imshow(raw_image)
plt.show()

# Test with a question
question = "Answer with organized answers: What type of rose is in the picture? Mention some of its characteristics and how to take care of it ?"
query_image = new_path + '/rose/8987479080_32ab912d10_n.jpg'
raw_image = Image.open(query_image)

doc = collection_text.query(
    query_embeddings = embedding_function(query_image), n_results = 1,
)['documents'][0][0]

plt.imshow(raw_image)
plt.show()
imgs = collection_images.query(query_uris = query_image, include = ['data'], n_results = 3)
for img in imgs['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

doc

prompt = """<|im_start|>system
A chat between a curious human and an artificial intelligence assistant.
The assistant is an exprt in flowers , and gives helpful, detailed, and polite answers to the human's questions.
The assistant does not hallucinate and pays very close attention to the details.<|im_end|>
<|im_start|>user
<image>
{question} Use the following article as an answer source. Do not write outside its scope unless you find your answer better {article} if you think your answer is better add it after document.<|im_end|>
<|im_start|>assistant
""".format(question = 'question', article = doc)

inputs = processor(prompt, raw_image, return_tensors = 'pt')
inputs['input_ids'] = inputs['input_ids'].to(model.device)
inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
inputs['pixel_values'] = inputs['pixel_values'].to(model.device)


streamer = TextStreamer(tokenizer)

output = model.generate(**inputs, 
                        max_new_tokens = 300, 
                        do_sample = True, 
                        top_p = 0.5, 
                        temperature = 0.2, 
                        eos_token_id = tokenizer.eos_token_id, 
                        streamer = streamer)
print(tokenizer.decode(output[0]).replace("<s> ", "").replace("</s>","").replace(prompt, "").replace("<|im_end|>", "").split('>assistant')[1])

# Summary
plt.imshow(raw_image)
plt.show()
imgs = collection_images.query(query_uris=query_image, include=['data'], n_results=3)
for img in imgs['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()
print('answer is ==> ' + 'The rose in the image is a beautiful pink flower with a purple background. It is surrounded by other purple flowers, creating a visually appealing scene. The rose is a woody perennial flowering plant of the genus Rosa, which belongs to the family Rosaceae. There are over three hundred species and tens of thousands of cultivars of roses, each with its own unique characteristics, such as size, shape, and color. The rose in the image is a beautiful pink flower with a purple background. It is surrounded by other purple flowers, creating a visually appealing scene.')