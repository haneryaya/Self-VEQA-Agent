#loop数为2

import json
import tqdm
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
import jsonlines
import re
from QA_Agent import task_specifier_agent
from Ver_Agent import veri_agent
from util import select_examples,select_reflexion_few_shot,calculate_simi_relation,select_reflexion_few_shot
from util import answer_retrieval_agent,get_ground_truth,get_que_hop_num,get_trainquestion,init,id_to_name,read_relationchain_and_question,name2id,calculate_simi_relation,get_que_hop_num,get_trainquestion
        


def main():
    question_chain_dic,id_name_dict = read_relationchain_and_question()
    question_hopnum_dic=get_que_hop_num()
    hop1_question_list,hop2_question_list=get_trainquestion()
    ground_truth_dic=get_ground_truth()
    entities = []#所有实体的名称
    with open('datasets/webqsp/entities.txt','r',encoding='utf-8') as f:
        for i in f.readlines():
            entities.append(i.replace('\n',''))
    relations = []#所有关系的名称
    with open('datasets/webqsp/relations.txt','r',encoding='utf-8') as f:
        for i in f.readlines():
            relations.append(i.replace('\n',''))

    with jsonlines.open('datasets/webqsp/test_simple.jsonl') as f:
        for line in tqdm.tqdm(f):
            ans = {}
            reflexion_result={}
            relation_chain=[]
            question = line['question']
            ans['id'] = question
            subgraph = line['subgraph']['tuples']#取当前问题的子图
            subgraph_relation=set()
            for sub in subgraph:
                subgraph_relation.add(relations[sub[1]])
            subgraph_relation=calculate_simi_relation(subgraph_relation,question)#返回个列表
            ans_dict,sub_relation,train_questions,train_inferencechain_dic,constraint_dict,constraints_name_dic, name_id_dic = init(entities,relations,subgraph)

            answer_ground_truth=ground_truth_dic[question]
            ans['ground_truth'] = answer_ground_truth
            with open("datasets/webqsp/question_no_head.json",encoding="utf-8") as f :
                contents=json.load(f)
                question_nohead=contents[question]
            



            hop_num=question_hopnum_dic[question]
            #加个对之前正确结果计算相似度的过程
            reflexion_few_shot=select_reflexion_few_shot(question_nohead,hop_num,0.8)
            examples=select_examples(question_nohead,hop1_question_list,hop2_question_list,train_inferencechain_dic,hop_num)#选择few-shot
            loop=2
            # inner_loop=5

            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2, max_tokens = 256) 
            summary_memory = ConversationBufferWindowMemory(llm=OpenAI(model_name='gpt-3.5-turbo'))
            #存plan memory
            conversation_plan = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=summary_memory)
            #存生成relation memory
            conversation_relation= ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=summary_memory)

            veification_response_sugg=''
            veification_response=''
            with_feedback=False
            begin = 0
            end = 10
            last_relation_lis=[]

            


            while loop:
                inner_loop=3
                while inner_loop:                    
                    plan_response=task_specifier_agent(question,conversation_plan,veification_response_sugg,with_feedback,examples[begin:end],last_relation_lis,hop_num,reflexion_few_shot)                  
                    matches = re.findall(r'\[([^\]]+)\]', plan_response)#按三个[]取出来的,是一个列表
                    if len(matches) == 0:
                        inner_loop-=1
                        continue
                    else:
                        break
                if len(matches) == 0:
                    relation_chain = []
                else:
                    relation_three_chains = matches#['kind 1','kind 2','kind 3']
                last_relation_lis=[]
                long_chain_lis=[]
                answer_name_lis=[]
                chain_current_num=0
                reflexion_dict={}
                for relation_chain in relation_three_chains:
                    last_relation_lis.append(relation_chain)
                    relation_chain=relation_chain.split(',')
                    i=0
                    long_chain=''
                    center_entity = [entities[i] for i in line['entities']]
                    long_chain+=id_to_name(center_entity[0],id_name_dict)+'->'
                    head=center_entity#现在我们有头了
                    ori_entity_id=head[0]
                    kg_relation_lis=[]
                    answer_name_id=''
                    answer=''
                    for r_chain in relation_chain:
                        r_chain=r_chain.strip()
                        long_chain+= r_chain +'->'
                        if i==0 and len(relation_chain)==1:
                            answer,kg_relation=answer_retrieval_agent(r_chain,True,head,ans_dict,ori_entity_id,sub_relation,False,conversation_relation)
                            kg_relation_lis.append(kg_relation)
                            answer_name_id=answer
                            answer=answer.split('; ')
                            for j in range(len(answer)):
                                answer[j]=id_to_name(answer[j],id_name_dict)
                            answer='; '.join(answer)
                            long_chain+= answer +'->'

                            i+=1
                        elif i==0 and len(relation_chain)>1:
                            answer,kg_relation=answer_retrieval_agent(r_chain,True,head,ans_dict,ori_entity_id,sub_relation,False,conversation_relation)
                            kg_relation_lis.append(kg_relation)
                            answer_name_id=answer
                            answer=answer.split('; ')
                            for j in range(len(answer)):
                                answer[j]=id_to_name(answer[j],id_name_dict)
                            answer='; '.join(answer)
                            long_chain+= answer +'->'


                            i+=1
                        else:
                            head=answer_name_id.split('; ')
                            answer,kg_relation=answer_retrieval_agent(r_chain,False,head,ans_dict,ori_entity_id,sub_relation,False,conversation_relation)
                            kg_relation_lis.append(kg_relation)
                            answer_name_id=answer
                            answer=answer.split('; ')
                            for j in range(len(answer)):
                                answer[j]=id_to_name(answer[j],id_name_dict)
                            answer='; '.join(answer)
                            long_chain+= answer +'->'

                    
                    answer_name=answer
                    if answer_name not in reflexion_dict:

                        reflexion_dict[answer_name]=[relation_chain]
                    answer_name_lis.append(['this is ' + str(chain_current_num) + ' kind of answer'])
                    answer_name_lis.append(answer_name)
                    long_chain_lis.append(['this is ' + str(chain_current_num) + ' kind of chain'])
                    long_chain_lis.append(long_chain)
                    chain_current_num+=1

                
                with open("datasets/webqsp/test_top200_reletriples_wsp.json",encoding="utf-8") as f :
                    contents=json.load(f)                    
                    relevant_triple_lis=contents[question]
                    relevant_triple_lis=relevant_triple_lis[:30]




                veification_response=veri_agent(answer_name_lis,conversation_relation,long_chain_lis,question,relevant_triple_lis)
                if 'success' not in veification_response:
                    loop-=1
                    break
                if '{' not in veification_response:
                    veification_response='{'+veification_response
                if '}' not in veification_response:
                    veification_response='{'+veification_response+'}'
                matches = re.findall(r'\{([^{}]*?(?:success)[^{}]*)\}', veification_response) 
                veification_response = '{'
                for i in matches:
                    if i!='\n':
                        veification_response += i
                veification_response += '}'   
                veification_response = json.loads(veification_response)
                veification_response_su=veification_response.get("success", "")

                answer_name_save=''
                veri_answer=''
                reflexion_result_relation=[]
                if isinstance(veification_response_su, str):
                    if veification_response_su.lower()=="true":
                        if answer_name==[None] or answer_name=='':
                            # answer_name_save="no answer"
                            veri_answer=veification_response.get("veri_answer", "")
                            if veri_answer==[None] or veri_answer=="no answer":
                                answer_name_save="no answer"
                            else:
                                answer_name_id_lis=[]
                                answer_name=veri_answer.split(';')

                                refle_answer_name=';'.join(answer_name)
                                if refle_answer_name in reflexion_dict:
                                    reflexion_result_relation=reflexion_dict[refle_answer_name]
                                    reflexion_result['question']=question
                                    reflexion_result['question_nohead']=question_nohead
                                    reflexion_result['relation']=reflexion_result_relation
                                    if hop_num=='1':
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)
                                    else:
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)


                                for i in range(len(answer_name)):
                                    answer_name_id_lis.append(name2id(answer_name[i].strip(),name_id_dic))
                                answer_name_id='; '.join(answer_name_id_lis)



                        else:
                            veri_answer=veification_response.get("veri_answer", "")
                            answer_name=veri_answer.split(';')
                            refle_answer_name=';'.join(answer_name)
                            if refle_answer_name in reflexion_dict:
                                reflexion_result_relation=reflexion_dict[refle_answer_name]
                                reflexion_result['question']=question
                                reflexion_result['question_nohead']=question_nohead
                                reflexion_result['relation']=reflexion_result_relation
                                if hop_num=='1':
                                    with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl', mode='a') as writer:
                                        writer.write(reflexion_result)
                                else:
                                    with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl', mode='a') as writer:
                                        writer.write(reflexion_result)
                            answer_name_id_lis=[]

                            answer_name=veri_answer.split(';')

                            for i in range(len(answer_name)):
                                answer_name_id_lis.append(name2id(answer_name[i].strip(),name_id_dic))
                            answer_name_id='; '.join(answer_name_id_lis)

                        break
                    else:
                        veification_response_sugg=veification_response.get("critique","")
                        with_feedback=True
                elif isinstance(veification_response_su, bool):
                    if veification_response_su==True:

                        if answer_name==[None] or answer_name=='':
                            # answer_name_save="no answer"
                            veri_answer=veification_response.get("veri_answer", "")
                            if veri_answer==[None] or veri_answer=="no answer":
                                answer_name_save="no answer"
                            else:
                                veri_answer=veification_response.get("veri_answer", "")
                                refle_answer_name=';'.join(answer_name)

                                if refle_answer_name in reflexion_dict:
                                    reflexion_result_relation=reflexion_dict[refle_answer_name]
                                    reflexion_result['question']=question
                                    reflexion_result['question_nohead']=question_nohead
                                    reflexion_result['relation']=reflexion_result_relation
                                    if hop_num=='1':
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)
                                    else:
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)                                        
                                answer_name_id_lis=[]

                                answer_name=veri_answer.split(';')

                                refle_answer_name=';'.join(answer_name)

                                if refle_answer_name in reflexion_dict:

                                    reflexion_result_relation=reflexion_dict[refle_answer_name]
                                    reflexion_result['question']=question
                                    reflexion_result['question_nohead']=question_nohead
                                    reflexion_result['relation']=reflexion_result_relation
                                    if hop_num=='1':
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)
                                    else:
                                        with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl', mode='a') as writer:
                                            writer.write(reflexion_result)





                                for i in range(len(answer_name)):
                                    answer_name_id_lis.append(name2id(answer_name[i].strip().replace('\'',''),name_id_dic))
                                answer_name_id='; '.join(answer_name_id_lis)

                        else:
                            veri_answer=veification_response.get("veri_answer", "")
                            answer_name=veri_answer.split(';')
                            refle_answer_name=';'.join(answer_name)

                            if refle_answer_name in reflexion_dict:
                                reflexion_result_relation=reflexion_dict[refle_answer_name]
                                reflexion_result['question']=question
                                reflexion_result['question_nohead']=question_nohead
                                reflexion_result['relation']=reflexion_result_relation
                                if hop_num=='1':
                                    with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_1hop_loop.jsonl', mode='a') as writer:
                                        writer.write(reflexion_result)
                                else:
                                    with jsonlines.open('datasets/webqsp/webqsp_reflexion_result_2hop_loop.jsonl', mode='a') as writer:
                                        writer.write(reflexion_result)

                            answer_name_id_lis=[]
                            answer_name=veri_answer.split(';')
                            for i in range(len(answer_name)):
                                answer_name_id_lis.append(name2id(answer_name[i].strip(),name_id_dic))
                            answer_name_id='; '.join(answer_name_id_lis)


                        break
                    else:
                        veification_response_sugg=veification_response.get("critique","")
                        with_feedback=True

                
                loop-=1
            if len(answer_name_id)==0:
                answer_name_id_string=veification_response.get("veri_answer", "")
                answer_name_id=answer_name_id_string.split(',')
                for i in range(len(answer_name_id)):
                    answer_name_id[i]=name2id(answer_name_id[i].strip(),name_id_dic)
                answer_name_id='; '.join(answer_name_id)


                    

            answer_name_id=answer_name_id.split('; ')
            if  ori_entity_id in answer_name_id:
                answer_name_id.remove(ori_entity_id)
            
            ans['prediction'] = answer_name_id
            with jsonlines.open('datasets/webqsp/output.jsonl', mode='a') as writer:
                writer.write(ans)

if __name__ == "__main__":
    print(main())

