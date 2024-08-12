import os
import json
import tqdm
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
import jsonlines
import re
from util import select_examples,answer_retrieval_agent,init,remove_brackets_and_content,get_relevant_triples
from QA_Agent import generate_agent,modify_agent
from Ver_Agent import veri_agent



def main():
    ans_dict = init()#字典嵌套字典
    # q='the films that share actors with the film [Dil Chahta Hai] were released in which years'
    # chain='starred_actors_reverse|release_year|release_year'
    # q='who are the directors of the movies written by the writer of [The Green Mile]'
    with open('datasets\\MetaQA\\2-hop\\qa_test.txt',encoding='utf-8') as f:
        contents=f.readlines()
        for content in tqdm.tqdm(contents):
            content=content.split('\t')
            cal_ans={}
            question=content[0]
            cal_ans['id'] = question
            ans=content[1].replace('\n','')
            cal_ans['ground_truth'] = ans
            
            

    # q='the films that share directors with the film [Catch Me If You Can] were in which languages'
            relation='release_year, written_by_reverse, directed_by_reverse, in_language, starred_actors, starred_actors_reverse, written_by, directed_by, has_genre'
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0, max_tokens = 256) 
            summary_memory = ConversationBufferWindowMemory(llm=OpenAI(model_name='gpt-3.5-turbo'))
            #存plan memory
            conversation_t = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=summary_memory) 
            #veri 记忆
            conversation_veri = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=summary_memory) 

            conversation_modify = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=summary_memory) 
            #初始化一条关系链
            loop=5
            inner_loop=3
            # while inner_loop: 
            question_nohead=remove_brackets_and_content(question)
            examples=select_examples(question_nohead)#20个，我们选5个
            begin=0
            end=10
            ge_resss=generate_agent(question,relation,conversation_t,examples[begin:end])


            ori_relation=re.findall(r'\[([^[\]]*?,[^[\]]*?)\]', ge_resss)
            print(ori_relation)
            if len(ori_relation)==0:
                ge_resss=generate_agent(question,relation,conversation_t,examples[begin:end])
                ori_relation = ori_relation[0].replace('"', '').split(',')

            else:
                ori_relation = ori_relation[0].replace('"', '').split(',')

                
            relevant_triples=get_relevant_triples(question)
            while loop:                

                head_entity=re.search(r'\[([^]]+)\]', question)
                head_entity=[head_entity.group(1)]
                origin_entity=head_entity
                is_first=True                   
                i=0 #是不是第一个关系
                full_chain=''
                full_chain+=str(head_entity[0])+'->'
                for relation_chai in ori_relation:  
                        full_chain+=relation_chai+'->'
                        if i==0:        
                            answer=answer_retrieval_agent(relation_chai,is_first,head_entity,ans_dict,origin_entity)
                            if answer=='no_this_relation':
                                full_chain+='no_this_relation'+'->'
                                break
                            else:
                                i+=1
                                is_first=False
                                head_entity=answer.split('; ')#head_entity是[]
                                full_chain+=answer+'->'
                        else :                        
                            answer=answer_retrieval_agent(relation_chai,is_first,head_entity,ans_dict,origin_entity)
                            if answer=='no_this_relation':
                                full_chain+='no_this_relation'+'->'
                                break
                            else:
                                answer_string=answer
                
                                answer=answer.strip().split('; ')#

                                if origin_entity[0] in answer:
                                    answer.remove(origin_entity[0])
                                head_entity=answer
                                answer_string='; '.join(answer)

                                full_chain+=answer_string+'->'


                veri_response=veri_agent(answer,conversation_veri,full_chain[:-2],question,relevant_triples) 
                if 'success' not in veri_response:
                    loop-=1
                    break
                if '{' not in veri_response:
                    veri_response='{'+veri_response+'}'
                matches = re.findall(r'\{([^{}]*?(?:success)[^{}]*)\}', veri_response) 
                veification_response = '{'
                for i in matches:
                    if i!='\n':
                        veification_response += i
                veification_response += '}'              
                veification_response = json.loads(veification_response)

                
                veification_response_su=veification_response.get("success", "")
                if isinstance(veification_response_su, str):
                    if veification_response_su.lower()=="true":
                        # print(answer)#还没存呢
                        loop=0
                        if answer==[None]:                            
                            answer_name_save="no answer"
                        else:
                            if origin_entity in answer:
                                answer.remove(origin_entity)


                        break
            
                    else:

                        modi_res=modify_agent(question,ori_relation,relation,conversation_modify,origin_entity,examples[0:5])
                        begin+=5
                        end+=5
                        modi_res=re.findall(r'\[([^[\]]*?,[^[\]]*?)\]', modi_res)
                        modi_res = modi_res[0].replace('"', '').split(',')
                        modi_res = [element.replace('"', '').replace("'", '') for element in modi_res]
                        
                        ori_relation=modi_res

                        
                elif isinstance(veification_response_su, bool):
                    if veification_response_su==True:
                        
                        loop=0
                        if answer==[None]:
                            answer_name_save="no answer"
                        else:
                            if origin_entity in answer:
                                answer.remove(origin_entity)



                        break
                    else:
                        modi_res=modify_agent(question,ori_relation,relation,conversation_modify,origin_entity,examples[0:5])
                        begin+=5
                        end+=5
                        modi_res=re.findall(r'\[([^[\]]*?,[^[\]]*?,[^[\]]*?)\]', modi_res)
                        modi_res = modi_res[0].replace('"', '').split(',')

                        modi_res = [element.replace('"', '').replace("'", '') for element in modi_res]
                        
                        ori_relation=modi_res

                loop-=1
            answer_lis=[]
            for a in answer:
                if a!=origin_entity[0]:
                    answer_lis.append(a)
            cal_ans['prediction']=answer_lis
            with jsonlines.open('datasets/output.jsonl', mode='a') as writer:
                writer.write(cal_ans)
                
    

if __name__ == "__main__":
    print(main())

        




