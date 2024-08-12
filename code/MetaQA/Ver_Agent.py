from util import openai_call_specifier
def veri_agent(answer_name,conversation_plan,chain,question,relevant_triples):
    prompt = f"""
Task:Here is the question:{question}. Judge whether a chain [{chain}] are reasonable to answer the question. pay more attention on whether relations in the chain are reasonable to answer the question.
Instruction: 
for relation with and without 'reverse', their meaning  are the same. pay less attention about 'reverse'.
If '' is in a chain or answer name is '', directly set "success" false.
If you think it is False, put your reasons in "critique".
There is no need to answer the question utilizing your internal knowledge.
Don't apology!
pay attention on the following example.
Note:
Your response must in JSON format as described below :
"success ": "True or False" ,
"critique":"why False"
Ensure the response can be parsed by Python ' json . loads ' , e . g .: no trailing commas , no single quotes , all contents should be strings etc .
Answers are {answer_name}. 
"""
    # print(f'\n****veri_agent AGENT PROMPT****\n{prompt}\n')
    response = openai_call_specifier(prompt,conversation=conversation_plan)    
    # print(f'\n****veri_agent AGENT RESPONSE****\n{response}\n')

    if not response:
        # print('Received empty response from veri_agent.')
        return
    return response