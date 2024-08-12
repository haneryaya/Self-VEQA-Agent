import os
import openai
import json
import time
import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import jsonlines
import re
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


os.environ['OPENAI_API_KEY']=''

LLM_MODEL = "gpt-3.5-turbo"
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
openai.api_key = OPENAI_API_KEY


def openai_call_specifier(
    prompt: str,
    conversation:ConversationChain
    
):
    loop = 3
    while loop:
        try:          
            return conversation.predict(input=prompt)           
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
            loop-=1
        else:
            break

# 选择前25个例子，分五次作为LLM解析关系链的few-shot
def select_examples(question):
    with open("datasets\\metaqa_relation_train_nohead.json",encoding="utf-8") as f :
        contents=json.load(f)
        train_questions=list(contents.keys())#[所有问题]
        train_inference_chains=[]#[所有关系链]
        for train_question in train_questions:
            train_inference_chain = contents[train_question]
            train_inference_chains.append(train_inference_chain)
    train_question_embeddings = np.load('datasets\\MetaQA\\2-hop\\train_question_nohead_emb.npy')#所有训练集问题的嵌入表示
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([question])
    cosine_sim = cosine_similarity(embeddings, train_question_embeddings)
    similarity_tri_question=sorted(zip(cosine_sim[0],train_questions,train_inference_chains),reverse=True)
    examples=[]
    for data in similarity_tri_question[:20]:
        raw_que=data[1]
        inference_chain=data[2]#[]
        if inference_chain==None:
            infe="[]"
        else:
            infe='['+','.join(inference_chain)+']'#字符串
        example="question:"+raw_que+"\ninference_chain:"+infe + '\n'
        examples.append(example)
    return examples
    
def answer_retrieval_agent(relation_list,is_first,extracted_head,ans_dict,origin_entity):

    relation_lis=relation_list.strip()
    center_entitys = extracted_head#头（中间）实体
    for i in center_entitys:
        i=i.strip()
    tmp_entitys = []
    for center_entity in center_entitys:

        if is_first or (not is_first and center_entity != origin_entity):
            if relation_lis in ans_dict[center_entity]:#判断当前关系是否在关系集中
                tmp_entity = ans_dict[center_entity][relation_lis]#头，关系对应的尾实体

                for i in tmp_entity:
                    tmp_entitys.append(i)

    if tmp_entitys!=[]:
        tmp_ans = '; '.join(set(tmp_entitys))
        return tmp_ans
    else: 
        if 'reverse' in relation_lis:
            relation_lis=relation_lis.replace('_reverse','')
        else:
            relation_lis=relation_lis+'_reverse'
            center_entitys = extracted_head
    for i in center_entitys:
        i=i.strip()
    tmp_entitys = []

    for center_entity in center_entitys:
        if is_first or (not is_first and center_entity != origin_entity):
            if relation_lis in ans_dict[center_entity]:
                tmp_entity = ans_dict[center_entity][relation_lis]

                for i in tmp_entity:
                    tmp_entitys.append(i)
    if tmp_entitys!=[]:
        tmp_ans = '; '.join(set(tmp_entitys))

        return tmp_ans
        
    return ''

def init():
    ans_dict = {}
    with open('datasets/2-hop/data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace('\"','')
            head,rel,tail = line.replace('\n','').split('\t')
            if head not in ans_dict:
                ans_dict[head] = {}
                ans_dict[head][rel] = [tail]
            else:
                if rel not in ans_dict[head]:
                    ans_dict[head][rel] = [tail]
                else:
                    ans_dict[head][rel].append(tail)
            
    ans_dict['']='no answers'
    return ans_dict


def cal_top50_simi_triple(two_hop_triples,question):
    two_hop_triples_lis=[]
    for i in two_hop_triples:
        two_hop_triples_lis.append(i)


    #嵌入每一个三元组，然后组成一个embedding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_embedding=[]
    embedding=[]
    for two_hop_triple in two_hop_triples:
        embedding.append(model.encode(two_hop_triple))
    all_embedding=[embedding]#全部三元组的嵌入
    #
    tokenizer_simcse = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model_simcse = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    inputs_simcse = tokenizer_simcse([question], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model_simcse(**inputs_simcse, output_hidden_states=True, return_dict=True).pooler_output[0]
    cosine_sim = cosine_similarity([embeddings], embedding)
    top50_triples = sorted(zip(cosine_sim[0],two_hop_triples_lis),reverse=True)
    top_50=[]
    for top50_triple in top50_triples:
        top_50.append(top50_triple[1])

    return top_50[:30]




def remove_brackets_and_content(input_string):
    # Define a regular expression pattern to match square brackets and their contents
    pattern = r'\[[^\]]*\]'

    # Use re.sub to replace matches with an empty string
    result = re.sub(pattern, '', input_string)

    return result

def get_relevant_triples(question):
    with open('datasets\\MetaQA\\2-hop\\test_2hop_top200_reletriples_metaqa.json') as f:
        contents=json.load(f)
        triples=contents[question]
        return triples[:30]
