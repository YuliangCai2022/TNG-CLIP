import torch
import openai
import time
import random
import openai
import asyncio
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def is_concrete_noun(word):
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    for syn in synsets:
        if syn.lexname() == "noun.artifact" or syn.lexname() == "noun.animal" or syn.lexname() == "noun.plant":
            return True
    return False

def extract_concrete_nouns(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    concrete_nouns = [word for word, tag in tagged_words if tag.startswith("NN")]
    return concrete_nouns
    
async def call_openai(prompt):
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

async def batch_openai(prompts):
    tasks = [call_openai(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

def prepair_similar_data(captions,image_id,image_feature):
    # output the captions [rule_align_negation1, llm_align_negation1,..... rule_contrast_negation1,llm_contrsat_negation1.....]
    # there're N image and 2N for align and 2N for contrast negation text
    # also extract the image feature from contrast image (1 per image) and concat with regular images
    random_list,similar_idx = get_similar_pair(captions,image_id,image_feature)
    object_list = get_negation_object(random_list)
    align_negation_captions = get_align_negation_caption(captions,object_list)
    contrast_negation_captions = get_contrast_negation_caption(captions)
    output_captions = []
    output_captions.extend(align_negation_captions)
    output_captions.extend(contrast_negation_captions)
    full_negation = []
    for idx in similar_idx:
        full_negation.append(contrast_negation_captions[0::2][idx])
    return output_captions, full_negation

def get_similar_pair(captions,image_id,image_feature):
    #input the normalized image_feature, find the closest image by cos similarity and form the pair
    prev_time = time.time()
    similarity_matrix = image_feature @ image_feature.T
    similarity_matrix.fill_diagonal_(-float('inf'))
    nearest_indices = similarity_matrix.argmax(dim=1)
    output_list = []
    for i in range(len(captions)):
        output_list.append([image_id[i],captions[i],image_id[nearest_indices[i]],captions[nearest_indices[i]]])
    
    return output_list, nearest_indices


def get_align_negation_caption(captions,objects):
    # align negation caption should contains 2 format: rule-based hybrid negation and diverse hybrid negation
    # returns one rule-based and one llm-based negatin for each of the caption (list of list) with rule based first and llm based second
    negation_patterns = [
        "{cap} with no {obj}.",
        "{cap} without {obj}",
        "{cap} that do not have {obj}.",
        "{cap} having no {obj}.",
        "{cap} not include {obj}.",
        "{cap} excluding {obj}.",
        "{cap}, but no {obj} are present.",
        "{cap}, though no {obj} can be seen.",
        "{cap} without any {obj} in sight.",
        "{cap} yet no {obj} are nearby.",
        "{cap} but no {obj} are visible.",
        "{cap} and no {obj} are anywhere around.",
        "{cap}, without any {obj} in the vicinity.",
        "{cap}, with no {obj} in the surroundings.",
        "{cap}, but no {obj} are in the area.",
        "{cap}, and no {obj} can be found nearby.",
        "{cap} in the absence of {obj}.",
        "{cap}, where no {obj} are present.",
        "{cap} with an absence of {obj}.",
        "{cap}, as no {obj} are around.",
        "{cap}, while lacking any {obj}.",
        "{cap} but no {obj} are engaging.",
        "{cap} with no {obj} participating.",
        "{cap} yet no {obj} are interacting.",
        "{cap}, as no {obj} are involved.",
        "{cap}, while {obj} remain absent from the scene.",
        "{cap} though no {obj} can be spotted.",
        "{cap} where no {obj} are noticeable.",
        "{cap} but no {obj} are detectable.",
        "{cap}, as no {obj} are apparent.",
        "{cap}, with no sight of any {obj}.",
        "No {obj} is visible, but {cap}.",
        "No {obj} can be seen, while {cap} happens.",
        "No {obj} is present, yet {cap} continues.",
        "No {obj} appears in sight, but {cap} unfolds.",
        "Not a single {obj} is noticeable, but {cap}.",
        "No trace of {obj} can be found, while {cap} occurs.",
        "No sign of {obj} is apparent, but {cap} is happening.",
        "There is no {obj} in view, but {cap} takes place.",
        "None of the {obj} are around, yet {cap} continues.",
        "Not even one {obj} is nearby, but {cap} is ongoing.",
        "No {obj} exists in the scene, while {cap} happens.",
        "Absolutely no {obj} is here, yet {cap} remains.",
        "Nowhere can {obj} be found, but {cap} is evident.",
        "Nowhere in sight is any {obj}, yet {cap} unfolds.",
        "No {obj} is around in the surroundings, but {cap} is occurring."
        
    ]

    def combine_negation(caption,obj,negation):
        caption = caption.split(".")[0]
        negation_caption= random.choice(negation).format(obj=obj, cap=caption)
        negation_caption = negation_caption.lower()
        return negation_caption.capitalize()

    rule_list = []
    llm_prepare_list = []
    output_list = []
    llm_list = []
    for i in range(len(captions)):
        rule_caption = combine_negation(captions[i],objects[i],negation_patterns)
        rule_list.append(rule_caption)
        #llm_prepare_list.append("rewrite the negation part of this sentence to make it make sense and more smooth: " + rule_caption + " The output sentence should have similar length than input sentence.")
        rule_caption = combine_negation(captions[i],objects[i],negation_patterns)
        llm_list.append(rule_caption)
    #llm_list = asyncio.run(batch_openai(llm_prepare_list))

    for i in range(len(rule_list)):
        output_list.append(rule_list[i])
        output_list.append(llm_list[i])
    return output_list


def get_contrast_negation_caption(captions):
    # contrast complete negation with the image itself, 2 format: rule based and diverse negation
    # return a list of list, [[rule_based, llm_based]]
    negation_words = ["There's no", "There is not", "The image does not include", "It is not shown that", "Not", "There does not exist", "There is nothing about"]
    negattion_pattern = [
        "There's no {cap} in the image.",
        "No {cap} is included in the image",
        "There is not {cap} in the image.",
        "The image does not have {cap}.",
        "No {cap} is present in the image.",
        "{cap} is not present in the image.",
        "{cap} is absent.",
        "No {cap} is present.",
        "There isn't any {cap}.",
        "Not a single {cap} can be seen.",
        "The image is without {cap}.",
        "The image is lacking {cap}.",
        "There appears to be no {cap} in the image.",
        "The image does not contain {cap}.",
        "There does not exsit {cap} in the image.",
        "There is nothing about {cap}.",
        "There isn't any {cap}.",
        "No {cap} is seen in the image."
    ]

    def combine_negation(caption,negation):
        caption = caption.split(".")[0]
        negation_caption= random.choice(negation).format(cap=caption)
        negation_caption = negation_caption.lower()
        return negation_caption.capitalize()
    
    rule_list = []
    llm_prepare = []
    llm_list = []
    for i in range(len(captions)):
        rule_list.append(combine_negation(captions[i],negattion_pattern))
        #llm_prepare.append("Rewrite this sentence to make it more smooth: " + random.choice(negation_words) + " " + captions[i].lower())
        llm_list.append(combine_negation(captions[i],negattion_pattern))

    #llm_list = asyncio.run(batch_openai(llm_prepare))


    output_list = []
    for i in range(len(captions)):
        output_list.append(rule_list[i])
        output_list.append(llm_list[i])
    return output_list


@torch.no_grad()
def get_negation_object(caption_list):

    input_text = []
    objects = []
    for i in range(len(caption_list)):
        found = False
        concrete_nouns = extract_concrete_nouns(caption_list[i][3])
        for noun in concrete_nouns:
            if noun not in caption_list[i][1]:
                objects.append(noun)
                found = True
        if not found:
            objects.append('butterfly')
    
    return objects


def save_model(model, path):
    torch.save(model.state_dict(), path)


