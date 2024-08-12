import os
import openai
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import numpy as np
import jsonlines
import json
import tqdm
import re
os.environ['OPENAI_API_KEY']='你的api_key'
LLM_MODEL = "gpt-3.5-turbo"
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
openai.api_key = OPENAI_API_KEY
def openai_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 2000,
):
    while True:
        try:
            if model.lower().startswith("text-"):
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message['content']
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break

#选择前25个例子，分五次作为LLM解析关系链的few-shot
def select_examples(question_nohead,hop1_question_list,hop2_question_list,train_inference_chain,hop_num):
    train_question_one_hop_embeddings = np.load('datasets/webqsp/embedding/embeddings_hop1_all-MiniLM-L6-v2.npy')#一跳嵌入位置
    train_question_two_hop_embeddings = np.load('datasets/webqsp/embedding/embeddings_hop2_all-MiniLM-L6-v2.npy')#二跳嵌入位置    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(question_nohead)
    examples=[]
    if hop_num=='1':
        cosine_sim = cosine_similarity([embeddings], train_question_one_hop_embeddings)
        datas = sorted(zip(cosine_sim[0],hop1_question_list),reverse=True)
        for data in datas[:50]:
            raw_que=data[1]
            inference_chain=train_inference_chain[raw_que]
            if inference_chain==None:
                infe="[]"
            else:
                infe='['+','.join(inference_chain)+']'
            
            example="question:"+raw_que+"\ninference_chain:"+infe + '\n'
            examples.append(example)
    if hop_num=='2':
        cosine_sim = cosine_similarity([embeddings], train_question_two_hop_embeddings)
        datas = sorted(zip(cosine_sim[0],hop2_question_list),reverse=True)
        for data in datas[:50]:
            raw_que=data[1]
            inference_chain=train_inference_chain[raw_que]
            if inference_chain==None:
                infe="[]"
            else:
                infe='['+','.join(inference_chain)+']'
            example="question:"+raw_que+"\ninference_chain:"+infe + '\n'
            examples.append(example)
    return examples


def answer_retrieval_agent(plan_response,is_first,extracted_head,ans_dict,origin_entity,sub_relation,re_generate_relation,conversation):
    relation_embeddings = np.load('datasets/webqsp/embedding/relations_embedding.npy')#关系嵌入路径
    sub_relation_embedding = []
    for i in sorted(sub_relation):
        sub_relation_embedding.append(relation_embeddings[i])
    

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(plan_response) #修改为直接对关系链进行嵌入，不用大模型生成关系了
    cosine_sim = cosine_similarity([embeddings], sub_relation_embedding)
    max_i=0
    current_index=0
    for i,cos in enumerate(cosine_sim[0]):
        if cos>=max_i:
            max_i=cos
            current_index=i
    relation = sub_relation[sorted(sub_relation)[current_index]]
    # print('KG relation:\t',relation)
    center_entitys = extracted_head
    # print('extracted_head:\t',center_entitys)
    tmp_entitys = []

    for center_entity in center_entitys:
        center_entity = center_entity.strip()
        if is_first or (not is_first and center_entity != origin_entity):
            if relation in ans_dict[center_entity]:
                tmp_entity = ans_dict[center_entity][relation]
                    # print(tmp_entity)
                for i in tmp_entity:
                    tmp_entitys.append(i)
        # print(tmp_entitys)
    if tmp_entitys!=[]:
        tmp_ans = '; '.join(set(tmp_entitys))
        return tmp_ans,relation
    return 'no answer',relation


def get_ground_truth():
    with open("datasets/webqsp/WebQSP.test.json",encoding="utf-8") as f :
        contents=json.load(f)
        ques=contents['Questions']
        ground_truth_dic={}
        for question in ques:
            current_ques_answer=''
            raw_question=question["ProcessedQuestion"]
            Answers=question["Parses"][0]["Answers"]
            for ans in Answers:
                current_ques_answer+=ans['AnswerArgument']+'|'

            ground_truth_dic[raw_question]=current_ques_answer
    return ground_truth_dic
 

def init(entity,relation,subgraph):
    with open("datasets/webqsp/WebQSP.train.json",encoding="utf-8") as f :
        contents=json.load(f)
        ques=contents['Questions']
        inferencechain_dic={}
        constrains_dic={}
        constraints_name_dic = {}
        questions=[]
        for question in ques:
            raw_question=question["ProcessedQuestion"]
            references_chain=question["Parses"][0]["InferentialChain"]

            questions.append(raw_question) 
            inferencechain_dic[raw_question]=references_chain
            constraints=question["Parses"][0]["Constraints"]
            const=[]
            const_name = []
            for constrain in constraints:
                if constrain["Operator"]=="Equal":
                    if constrain["EntityName"]:
                        const.append(constrain["NodePredicate"])
                        const_name.append(constrain["EntityName"])
                    else:
                        const.append(constrain["NodePredicate"])
                        const_name.append(constrain["Argument"])
            constrains_dic[raw_question]=const
            constraints_name_dic[raw_question] = const_name

    ans_dict = {}#
    sub_relation = {}#子图关系字典
    for head,rel,tail in subgraph:#遍历子图里三元组的三个部分
        sub_relation[rel] = relation[rel]#把子图中的relation取出来
        head = entity[head]
        tail = entity[tail]
        rel = relation[rel]
        if head not in ans_dict:
                ans_dict[head] = {}
                ans_dict[head][rel] = [tail]
        else:
            if rel not in ans_dict[head]:
                ans_dict[head][rel] = [tail]
            else:
                ans_dict[head][rel].append(tail)
        
        if tail not in ans_dict:
            ans_dict[tail] = {}
            ans_dict[tail][rel] = [head]
        else:
            if rel not in ans_dict[tail]:
                ans_dict[tail][rel] = [head]
            else:
                ans_dict[tail][rel].append(head)
    ans_dict['no answer']=''
    ans_dict['']='no answer'
    name_id_dic = {}
    with open('datasets/webqsp/name2id.txt','r',encoding='utf-8') as f:
        for content in f.readlines():
            i = content.split('\t')
            name_id_dic[i[0].lower()] = i[1]
            

    return ans_dict,sub_relation,questions,inferencechain_dic,constrains_dic,constraints_name_dic,name_id_dic

def id_to_name(entity_id,id_name_dict):
    if entity_id=='no answer':
        return ''
    if entity_id not in id_name_dict:
        return entity_id
    return id_name_dict[entity_id]
                 
def extract_content(input_string):
    pattern = r'\{([^}]+)\}'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def read_relationchain_and_question():
    with open("datasets/webqsp/WebQSP.test.json",encoding="utf-8") as f :
        contents=json.load(f)
        ques=contents['Questions']
        inferencechain_dic={}
        questions=[]
        for question in ques:
            raw_question=question["ProcessedQuestion"]
            references_chain=question["Parses"][0]["InferentialChain"]
            questions.append(raw_question) 
            inferencechain_dic[raw_question]=references_chain if references_chain else []

    with open("datasets/webqsp/entity_list_file_freebase_complete_all_mention_modification",encoding='utf-8') as f:
        contents=f.readlines()
        id_name_dict = {}
        for content in contents:
            entity_id = content.split('\t')[0]
            content_name=content.split('\t')[1]
            id_name_dict[entity_id] = content_name
    return inferencechain_dic,id_name_dict

def select_intersection_using_constraint(conversation_relation,sub_relation,ans_dict,center_entitys,origin_entity,relation_chain,constraint_set,location):
    flag = True
    prompt_feedback=False
    kg_relation_lis=[]
    center_entity = center_entitys
    middle_answer_loc1 = []
    middle_answer_loc0 = []
    first = True
    for plan_response in relation_chain:
        answer,kg_relation=answer_retrieval_agent(plan_response,flag,center_entity,ans_dict,origin_entity,sub_relation,prompt_feedback,conversation_relation)
        kg_relation_lis.append(kg_relation)
        center_entity = answer.split('; ')
        if first:
            middle_answer_loc0 = center_entity
            first = False
        if location==1:
            center_entity = list(set(center_entity).intersection(constraint_set))
            middle_answer_loc1 = center_entity
            location = -1
        
        flag = False
    
    if location==0:
        print('-------------------center_entity------------------\n',center_entity)
        return list(set(center_entity).intersection(constraint_set)),middle_answer_loc0
    return center_entity,middle_answer_loc1

def name2id(name,name_id_dic):
    name=name.lower()
    if name not in name_id_dic:
        return name
    return name_id_dic[name].replace('\n','')

def calculate_simi_relation(sub_relation,question):
    sub_relation_lis=[]
    for sub_rel in sub_relation:
        sub_relation_lis.append(sub_rel)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sub_relation_embedding = model.encode(sub_relation_lis)
    embeddings = model.encode(question) #修改为直接对关系链进行嵌入，不用大模型生成关系了
 #修改为直接对关系链进行嵌入，不用大模型生成关系了
    cosine_sim = cosine_similarity([embeddings], sub_relation_embedding)
    datas = sorted(zip(cosine_sim[0],sub_relation_lis),reverse=True)
    simi_relations=[]
    for data in datas[:20]:
        simi_relations.append(data[1])
    return simi_relations
        
def get_que_hop_num():
    with open('datasets/webqsp/hop_num_test.json',encoding='UTF-8') as f:
        hop_dict=json.load(f)
    return hop_dict

def get_trainquestion():
    with open("datasets/webqsp/WebQSP.train.json",encoding="utf-8") as f :
        contents=json.load(f)
        hop_1=[]
        hop_2=[]
        for cont in contents:
            raw_question=cont["question"]
            hop_num=cont["hop"]
            if hop_num==1:
                hop_1.append(raw_question)
            if hop_num==2:
                hop_2.append(raw_question)
    return hop_1,hop_2

def select_reflexion_few_shot(question_nohead,hop_num,threshold):
    if hop_num=='1':
        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl') as f:
            question_list=[]
            question_nohead_list=[]
            relation_list=[]
            is_empty = True
            for line in tqdm.tqdm(f):
                is_empty = False
                question_list.append(line['question'])
                question_nohead_list.append(line['question_nohead'])
                for i in line['relation']:
                    if i not in relation_list:
                        relation_list.append(i)
    if hop_num=='2':
        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl') as f:
            question_list=[]
            question_nohead_list=[]
            relation_list=[]
            is_empty = True
            for line in tqdm.tqdm(f):
                is_empty = False
                question_list.append(line['question'])
                question_nohead_list.append(line['question_nohead'])
                for i in line['relation']:
                    if i not in relation_list:
                        relation_list.append(i)
    if is_empty:
        return []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    raw_question_embeddings = model.encode(question_nohead)
    few_shot_embeddings = model.encode(question_nohead_list)
    cosine_sim = cosine_similarity([raw_question_embeddings], few_shot_embeddings)
    datas = sorted(zip(cosine_sim[0],question_list,relation_list,question_nohead_list),reverse=True)
    few_shot_examples=[]
    for data in datas[:3]:
        if data[0]>threshold:
            relation_chain=','.join(data[2])
            example="question:"+data[1]+"\n inference_chain:"+relation_chain + '\n'
            few_shot_examples.append(example)
    return few_shot_examples

