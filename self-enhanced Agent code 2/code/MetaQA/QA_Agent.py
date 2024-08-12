from util import openai_call_specifier
def generate_agent(question,relations,conversation,examples):
    prompt=f'''
task: 
you need give me a inference chain that made of two relations in order by imitating the examples according to the following question.
Here is the target question: {question} 
instruction:
the chain is composed of relations which can be used to answer the question. 
pay more attention on the orders of relations which is important. 
the two relations should be choosen from [{relations}]
Here are examples:{examples}
output: strictly follow the format: inference chain:[relation1,relation2]
note:only return inference chain containing two relations, no need to explain
please don't apologize!


'''
    response = openai_call_specifier(prompt,conversation=conversation)
    print('generate_agent***response',response)
    return response

def modify_agent(question,chain,relations,conversation,center_entity,examples):
    prompt=f'''
    task:you need to modify the chain by imitating examples. new chain must contain two relations
    instruction:the chain is composed of relations which can used to answer the question.
    the two relations should be selectes from :{relations}
    question:{question}
    center_entity:{center_entity}
    note:only return a new chain,the new chain should only contain two relations. pay more attention on the orders of relations. the first relation should point to center_entity. last relation must point to answer.
    don't generate the same one as {chain}
    output: strictly follow the format: inference chain:[relation1,relation2]
    Here are examples:{examples}
    '''
    response = openai_call_specifier(prompt,conversation=conversation)
    with open("C:\\Users\\WangMengHan\\Desktop\\metaqa_2hop.txt", "a",encoding="utf-8") as file:
        file.write("\n\n\n\n\n"+prompt+"****modi RESPONSE****"+"\n"+response + "\n")
    return response
