import pandas as pd
with open('samples.md', 'w', encoding='utf-8') as f:
    df_url=pd.read_csv('train_xgboost/data/urldata.csv')
    f.write('### 🔗 3 SAFE URLs\n')
    for u in df_url[df_url['label']==0]['url'].head(3): f.write(f'- {u}\n')
    f.write('\n### 🔗 3 PHISHING URLs\n')
    for u in df_url[df_url['label']==1]['url'].head(3): f.write(f'- {u}\n')
    
    df_email=pd.read_csv('train_roberta/data/Phishing_Email.csv')
    f.write('\n### 📧 3 SAFE EMAILS\n')
    for i, e in enumerate(df_email[df_email['Email Type']=='Safe Email']['Email Text'].dropna().head(3)):
        f.write(f'**Safe Email {i+1}:**\n```\n{e[:800]}...\n```\n\n')
    f.write('\n### 📧 3 PHISHING EMAILS\n')
    for i, e in enumerate(df_email[df_email['Email Type']=='Phishing Email']['Email Text'].dropna().head(3)):
        f.write(f'**Phishing Email {i+1}:**\n```\n{e[:800]}...\n```\n\n')
