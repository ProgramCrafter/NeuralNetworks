import requests
import json
import time

try:
  with open(__file__ + '/../cf-submissions.json') as f:
    submissions = json.load(f)
except:
  submissions = []

url_submissions = 'https://codeforces.com/api/contest.status?contestId=%d&from=1&count=10000'

added_submissions = 0

for contest in [1677, 1678, 1556, 1558, 1560, 1548, 1552]:
  time.sleep(0.4)
  new_submissions = requests.get(url_submissions % contest).json()
  
  if new_submissions['status'] == 'OK':
    for submission in new_submissions['result']:
      if 'verdict' not in submission:
        print('NO VERDICT IN SUBMISSION', submission)
        continue
      
      if submission['verdict'] != 'OK': continue
      if 'rating' not in submission['problem']: continue
      
      if submission.get('relativeTimeSeconds', 2**31-1) == 2**31-1: continue
      
      submissions.append([
        submission['problem']['rating'],
        submission['programmingLanguage'],
        submission['timeConsumedMillis'],
        submission['memoryConsumedBytes'],
        submission['relativeTimeSeconds']
      ])
      added_submissions += 1
    
    with open(__file__ + '/../cf-submissions.json', 'w') as f:
      json.dump(submissions, f, indent=2)
  else:
    print(new_submissions)

print(added_submissions)
