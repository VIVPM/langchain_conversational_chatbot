from supabase import create_client
from dotenv import load_dotenv
import re
import json
from langchain.evaluation.qa import QAEvalChain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import SambaNovaCloud
load_dotenv()
import os
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

llm = SambaNovaCloud(model='Meta-Llama-3.1-8B-Instruct', sambanova_api_key=os.getenv('SAMBANOVA_API_KEY'))
prompt = PromptTemplate.from_template("Answer the following question: {query}")
qa = LLMChain(llm=llm, prompt=prompt)

response = supabase.table("profiles").select("chats").execute()

all_chats = []
for profile in response.data:
    if 'chats' in profile:
        all_chats.extend(profile['chats'])

examples = []
predictions = []

for chat in all_chats:
    messages = chat['messages']
    i = 0
    while i < len(messages):
        if messages[i]['role'] == 'user':
            query = messages[i]['content']
            i += 1
            if i < len(messages) and messages[i]['role'] == 'assistant':
                result = messages[i]['content']
                examples.append({'query': query})
                result = result.strip('**Answer:**\n')
                predictions.append({'result':result})
                i += 1
            else:
                i += 1
        else:
            i += 1
            
with open('datasets/questions.json','w') as f:
    f.write(json.dumps(examples))
    
ground_truths = qa.apply(examples)

examples_for_eval = []
for i in range(len(examples)):
    q = examples[i]['query']
    gt = ground_truths[i]['text']
    examples_for_eval.append({'query': q, 'answer': gt})

with open('datasets/dataset.json','w') as f:
    f.write(json.dumps(examples_for_eval))

custom_prompt = ChatPromptTemplate.from_template(
    "You are an expert scorer. Given the question, the correct answer, and the predicted answer, give a score from 1 to 10 on how well the predicted answer matches the correct answer in terms of accuracy, completeness, and relevance.\n"
    "Score 1: Completely wrong or irrelevant\n"
    "Score 10: Perfect match or equivalent\n"
    "Question: {query}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {result}\n"
    "Score:"
)

eval_chain = QAEvalChain.from_llm(llm, prompt=custom_prompt)
graded_outputs = eval_chain.evaluate(examples_for_eval, predictions)

scores=[]
for i, eg in enumerate(examples_for_eval):
    print(f"Example {i}:")
    print("Question: " + eg['query'])
    print("Real Answer: " + eg['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    results = graded_outputs[i]['results']
    match = re.search(r'Score: (\d+)', results)
    if match:
        scores.append(int(match.group(1)))
    print()
average = sum(scores) / len(scores) if scores else 0
print(f"Average Score: {round(average*10,2)}")