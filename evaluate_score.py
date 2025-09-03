from supabase import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path="../crewai/.env")
import os
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)
score = 0
count = 0
response = supabase.table("profiles").select("chats").execute()
all_chats = [chat for user in response.data for chat in user["chats"]]
for i in range(len(all_chats)):
    for j in range(len(all_chats[i]["messages"])):
        if all_chats[i]['messages'][j]['role'] == 'assistant':
            if 'score' in all_chats[i]['messages'][j]:  
                score += all_chats[i]['messages'][j]['score']
                count += 1
print('Accuarcy:',round((score/count),2)/10*100)
                