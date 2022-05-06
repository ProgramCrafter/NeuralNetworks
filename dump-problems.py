from data_source import CFProblemTimingsDataSource

import sys

values_by_columns = [[], [], [], [], []]

def log_row(lang, time, memory, subtime, rating):
  values_by_columns[0].append(lang)
  values_by_columns[1].append(time)
  values_by_columns[2].append(memory)
  values_by_columns[3].append(subtime)
  values_by_columns[4].append(rating)

dataset = CFProblemTimingsDataSource(__file__ + '/../cf-submissions.json')
for i in range(dataset.cases()):
  lang, time, memory, subtime, _ = dataset.extract_data(i)
  rating, = dataset.wanted(i)
  
  log_row(lang, time, memory, subtime, rating)
  print('\t%s\t%.3f\t%.3f\t%.3f   ->   %.3f' % (lang, time, memory, subtime, rating), file=sys.stderr)

min_by_columns = [min(a) for a in values_by_columns]
max_by_columns = [max(a) for a in values_by_columns]
avg_by_columns = [sum(a) / len(a) for a in values_by_columns]
med_by_columns = [sorted(a)[len(a)//2] for a in values_by_columns]

print('\nCOUNT:', dataset.cases())
print()
print('MIN\t%.3f\t%.3f\t%.3f\t%.3f   ->   %.3f' % tuple(min_by_columns))
print('AVG\t%.3f\t%.3f\t%.3f\t%.3f   ->   %.3f' % tuple(avg_by_columns))
print('MED\t%.3f\t%.3f\t%.3f\t%.3f   ->   %.3f' % tuple(med_by_columns))
print('MAX\t%.3f\t%.3f\t%.3f\t%.3f   ->   %.3f' % tuple(max_by_columns))
