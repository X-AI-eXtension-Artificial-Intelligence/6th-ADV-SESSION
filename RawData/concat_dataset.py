import pandas as pd

## 1. SNS 데이터
## topic, keyword, query, response, response_speaker, response_sex, response_age
df1 = pd.read_csv('Dataset/utterance_sns.csv')

## 2. 주제별 일상 대화 데이터
## query, response, response_sex, response_age, mediatype, medianame, subject
df2 = pd.read_csv('Dataset/utterance_topic.csv')

## 3. 페르소나 데이터
## topic, query, response, response_persona_profiles, response_persona_profile_majors, response_persona_profile_minors
df3 = pd.read_csv('Dataset/utterance_persona.csv')

## 통합 데이터 컬럼
## query, response, label(성별), age, topic
df1 = df1[['query', 'response', 'response_sex', 'response_age', 'topic']]
df1.response_sex = df1.response_sex.apply(lambda x : '남성' if x == '남자' else '여성')
df1.response_age = df1.response_age.apply(lambda x : str(x)+'대')
df1.columns = ['query', 'response', 'gender', 'age', 'topic']

df2 = df2[['query', 'response', 'response_sex', 'response_age', 'subject']]
df2.columns = ['query', 'response', 'gender', 'age', 'topic']

df3 = df3[df3.response_persona_profile_majors.str.contains('성별') & df3.response_persona_profile_majors.str.contains('연령')]
df3['gender'] = df3.response_persona_profiles.apply(lambda x : '여성' if '나는 여자다' in x else '남성')
df3['age'] = df3.response_persona_profiles.apply(lambda x : '10대' if '10대' in x.split('|')[1] else '20대' if '20대' in x.split('|')[1] else '30대' if '30대' in x.split('|')[1] else '40대'
                                                 if '40대' in x.split('|')[1] else '50대' if '50대' in x.split('|')[1] else '60대')
df3 = df3[['query', 'response', 'gender', 'age', 'topic']]

total_df = pd.concat([df1, df2, df3], axis = 0)
total_df.to_csv('Dataset/utterance_total.csv', index = False)