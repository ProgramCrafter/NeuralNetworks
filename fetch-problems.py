import requests
import json

try:
  with open(__file__ + '/../cf-submissions.json') as f:
    submissions = json.load(f)
except:
  submissions = []

url_submissions = 'https://codeforces.com/api/problemset.recentStatus?count=999'
# url_submissions = 'https://codeforces.com/api/user.status?handle=TeaPot&from=1&count=10000'
new_submissions = requests.get(url_submissions).json()

if new_submissions['status'] == 'OK':
  for submission in new_submissions['result']:
    if 'verdict' not in submission:
      print('NO VERDICT IN SUBMISSION', submission)
      continue
    
    if submission['verdict'] != 'OK': continue
    if 'rating' not in submission['problem']: continue
    
    submissions.append([
      submission['problem']['rating'],
      submission['programmingLanguage'],
      submission['timeConsumedMillis'],
      submission['memoryConsumedBytes']
    ])
  
  with open(__file__ + '/../cf-submissions.json', 'w') as f:
    json.dump(submissions, f, indent=2)
else:
  print(new_submissions)
