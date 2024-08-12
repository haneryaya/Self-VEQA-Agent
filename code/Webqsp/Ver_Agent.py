from util import openai_call
def veri_agent(answer_name_lis,conversation_plan,chain_lis,question,fact_triples):
    # chain_lis=chain_lis.replace('{','(').replace('}',')')
    prompt = f"""
Task:Judge whether one of these chains {chain_lis} can answer the question and pick one of a kind answer as 'veri_answer' to return and set 'success' true.
Instruction: 
Pay more attention on the relations of the chains when you pick answers.
If Answers are all empty, you should utilizing triples to answer the question if possible and put only answer in "veri_response" without any uselessword such as 'and' ,and just use semicolon to seperate different answers, otherwise directly set "success" false.
There is no need to answer the question utilizing your internal knowledge.
But you can judge whether the answer is correct to answer the question utilizing your internal knowledge.
Don't apology!
Note:
Your response must in JSON format as described below :
" success ": "True or False" ,
" veri_answer ": "one kind answer or answer it utilizing KnowledgeGraph" ,
" critique ": " critique " ,
Ensure the response can be parsed by Python ' json . loads ' , e . g .: no trailing commas , no single quotes , all contents should be strings etc .
Pay more attention on the output format, which is important, otherwise,it will influence latter work.
When you pick one kind of answers from different kinds of Answers, you should choose all answers from only one kind you pick
KnowledgeGraph: Here are fact triples used to answer the question.{fact_triples}
question:{question}
different kinds of Answers are {answer_name_lis}. 

"""
    # prcess_pro=prompt.split(' ')
    if len(prompt)>=15000:
        prompt=prompt[:9000]
    print(f'\n****veri_agent AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt)
    print(f'\n****veri_agent AGENT RESPONSE****\n{response}\n')

    if not response:
        print('Received empty response from veri_agent.')
        return
    return response