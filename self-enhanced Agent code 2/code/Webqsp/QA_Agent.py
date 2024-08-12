from util import openai_call
from langchain.chains import ConversationChain
def task_specifier_agent(question,conversation:ConversationChain,verification_response,with_feedback,examples,last_response,hop_num,reflexion_few_shot):
    # Define the base instruction for the task
    instruction = f"""
    TASK: Generate two kinds of the most reasonable inference chains by imitating the provided examples. The number of relations should be {hop_num}.
    Instruction: 
    Pay more attention to the examples {examples}.
    """

    # Add reflexion few-shot examples if available
    if reflexion_few_shot:
        instruction += f"Here are several most similar examples to your question which you can infer to {reflexion_few_shot}.\n"

    # Modify the instruction based on feedback status
    if with_feedback:
        instruction += f"""
        according to critiques {verification_response}. 
        Don't generate the same inference chain as {last_response}.
        """

    # Add the final part of the prompt based on hop_num
    instruction += f"""
    You must give me two different chains.
    Question: {question}
    The number of relations is important; pay more attention to it according to the information provided to you: {hop_num} relation(s).
    Note: Output only two inference chains containing {hop_num} relation(s), following strictly the output format:
    
    """

    if hop_num == '2':
        instruction += "Inference chain1:[relation3, relation4].Inference chain2:[relation1, relation2].Inference chain3:[relation1, relation2]"
    elif hop_num == '1':
        instruction += "Inference chain1:[relation1].Inference chain2:[relation2].Inference chain3:[relation3]."
    
    # Make the OpenAI call
    response = openai_call(instruction)
    
    # Print the response for debugging
    print(f'\n****TASK specifier AGENT RESPONSE****\n{response}\n')
    
    # Handle empty response
    if not response:
        print('Received empty response from specifier agent.')
        return
    
    return response
