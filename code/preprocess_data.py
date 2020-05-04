'''
Preprocess the Hong Kong horse racing dataset and
create training, validation and test splits
'''

from xlrd import open_workbook
import sys
import numpy as np
import sklearn
from sklearn.svm import SVC
import random
import itertools
from sklearn import preprocessing
from collections import Counter

# utility functions
def read_horse_details():
  # reads race-result-horse.csv file containing details of horse
  headers = None
  horse_details = {}
  for line in open('data/raw/race-result-horse.csv'):
    if not headers:
      headers = line.strip().split(",")
      continue
    items = line.strip().split(',')
    values = {}
    for item, header in zip(items, headers):
      values[header] = item
    race_id, finishing_position = values['race_id'], values['finishing_position']
    if race_id not in horse_details:
      horse_details[race_id] = {}
    horse_details[race_id][finishing_position] = values 
  return horse_details

def read_race_details():
  # reads race-result-race.xlsx file containing details of race
  race_details = {}
  rb = open_workbook('data/raw/race-result-race.xlsx')
  r_sheet = rb.sheet_by_index(0)
  for ri in range(1, 2368):
    values = {}
    for ci in range(12):
      values[r_sheet.cell(0, ci).value] = r_sheet.cell(ri, ci).value
    race_details[values['race_id']] = values
  return race_details

def get_vocab(details, vocab_size):
  # creates word vocabulary mapping from word to id and id to word
  # from match summary
  vocab_count = Counter()
  for d in details:
    detail = details[d]
    for w in detail['incident_report'].split():
      vocab_count[w] += 1
  word2id, id2word = {}, {}
  for w,c in vocab_count.most_common()[0:vocab_size]:
    word2id[w] = len(id2word)
    id2word[word2id[w]] = w
  print('vocab. size = %d'%len(word2id))
  return word2id, id2word

def get_mean_horse(feat):
  # get mean value for a particular horse feature
  num, su = 0.0, 0.0
  for race_id in horse_details:
    for horse_id in horse_details[race_id]:
      if horse_details[race_id][horse_id][feat].isdigit():
        num += 1.0
        su += float(horse_details[race_id][horse_id][feat])
  return su/num

def time_to_100thseconds(time):
  # convert time in mm:ss:100ths format to 100ths
  m, s, ms = time.split('.')
  ms_a = int(m) * 60 * 100 + int(s) * 100 + int(ms)
  return ms_a

def get_local_wordfreq(report):
  # convert a sentence into word to frequency mapping
  words = {}
  for w in report.split():
    if w not in words:
      words[w] = 0
    words[w] = words[w] + 1
  return words

def read_Xy(race_details, horse_details, user_feats, word2id, id2word):
  # create the features and label for each horse
  race2Xy = {}
  jockey2id, id2jockey = {}, {}
  trainer2id, id2trainer = {}, {}
  raceclass2id, id2raceclass = {}, {}
  racecourse2id, id2racecourse = {}, {}
  trackcond2id, id2trackcond = {}, {}
  racename2id, id2racename = {}, {}
  track2id, id2track = {}, {}
  for race_id in race_details:
    race2Xy[race_id] = []
    for horse_id in horse_details[race_id]:
      x, y = [], None
      race_info = race_details[race_id]
      horse_info = horse_details[race_id][horse_id]
      # add the label
      if horse_info['finish_time'] == "---" or not horse_info['finishing_position'].isdigit():
        continue
      y = time_to_100thseconds(horse_info['finish_time'])

      # add horse details
      # add horse number
      if 'horse_number' in user_feats:
        if horse_info['horse_number'].isdigit():
          x.append(int(horse_info['horse_number']))
        else:
          x.append(0)
      
      # add jockey id
      if 'jockey' in user_feats:
        if horse_info['jockey'] not in jockey2id:
          jockey2id[horse_info['jockey']] = len(id2jockey)
          id2jockey[jockey2id[horse_info['jockey']]] = horse_info['jockey']
        x.append(jockey2id[horse_info['jockey']])

      # add trainer id
      if 'trainer' in user_feats:
        if horse_info['trainer'] not in trainer2id:
          trainer2id[horse_info['trainer']] = len(id2trainer)
          id2trainer[trainer2id[horse_info['trainer']]] = horse_info['trainer']
        x.append(trainer2id[horse_info['trainer']])
      
      # add horse weights
      if 'actual_weight' in user_feats:
        if horse_info['actual_weight'].isdigit():
          x.append(int(horse_info['actual_weight']))
        else:
          x.append(mean_actual_wt)
      
      if 'declared_horse_weight' in user_feats:
        if horse_info['declared_horse_weight'].isdigit():
          x.append(int(horse_info['declared_horse_weight']))
        else:
          x.append(mean_declr_wt)

      # add draw
      if 'draw' in user_feats:
        if horse_info['draw'].isdigit():
          x.append(int(horse_info['draw']))
        else:
          x.append(10)

      # add race details
      # add race course
      if 'race_course' in user_feats:
        if race_info['race_course'] not in racecourse2id:
          racecourse2id[race_info['race_course']] = len(id2racecourse)
          id2racecourse[racecourse2id[race_info['race_course']]] = race_info['race_course']
        x.append(racecourse2id[race_info['race_course']])

      # add race number
      if 'race_number' in user_feats:
        x.append(int(race_info['race_number']))

      # add race class
      if 'race_class' in user_feats:
        if race_info['race_class'] not in raceclass2id:
          raceclass2id[race_info['race_class']] = len(id2raceclass)
          id2raceclass[raceclass2id[race_info['race_class']]] = race_info['race_class']
        x.append(raceclass2id[race_info['race_class']])

      # add race distance
      if 'race_distance' in user_feats:
        x.append(int(race_info['race_distance']))
      
      # add track condition
      if 'track_condition' in user_feats:
        if race_info['track_condition'] not in trackcond2id:
          trackcond2id[race_info['track_condition']] = len(id2trackcond)
          id2trackcond[trackcond2id[race_info['track_condition']]] = race_info['track_condition']
        x.append(trackcond2id[race_info['track_condition']])

      # add race name
      if 'race_name' in user_feats:
        if race_info['race_name'] not in racename2id:
          racename2id[race_info['race_name']] = len(id2racename)
          id2racename[racename2id[race_info['race_name']]] = race_info['race_name']
        x.append(racename2id[race_info['race_name']])

      # add track id
      if 'track' in user_feats:
        if race_info['track'] not in track2id:
          track2id[race_info['track']] = len(id2track)
          id2track[track2id[race_info['track']]] = race_info['track']
        x.append(track2id[race_info['track']])

      # add text query
      cur_words = get_local_wordfreq(race_info['incident_report'])
      for wi in range(len(id2word)):
        if id2word[wi] in cur_words:
          x.append(cur_words[id2word[wi]])
        else:
          x.append(0)

      race2Xy[race_id].append((horse_id, x, y))
  return race2Xy

def create_matrix(races, race2Xy):
  # create the numpy matrix for features and labels
  Xy_pairs = []
  for race in races:
    for horse in race2Xy[race]:
      Xy_pairs.append((horse[1], horse[2]))
  random.shuffle(Xy_pairs)
  X, y = [], []
  for pair in Xy_pairs:
    X.append(pair[0])
    y.append(pair[1])
  X = np.array(X, dtype=np.float32)
  y = np.array(y, dtype=np.int)
  return X, y

if __name__== "__main__":
  # process the command line arguments
  if len(sys.argv) != 3:
    print("error: incorrect inputs to the program")
    print("format: python preprocess_data.py all 1487")
    sys.exit(0) 
  user_feats = None
  if sys.argv[1] == "all":
    user_feats = ','.join(['horse_number', 'jockey', 'trainer', 'actual_weight', 'declared_horse_weight', 'draw', 'race_course', 'race_number', 'race_class', 'race_distance', 'track_condition', 'race_name', 'track'])
  else:
    user_feats = sys.argv[1].split(",")
  vocab_size = int(sys.argv[2])

  # set seed for pseudo random generator
  random.seed(123)
  
  # read horse details
  horse_details = read_horse_details()

  # read race details
  race_details = read_race_details()

  # create word vocabulary from match summary
  word2id, id2word = get_vocab(race_details, vocab_size)

  # create mean values for some horse features
  mean_declr_wt = int(get_mean_horse('declared_horse_weight'))
  mean_actual_wt = int(get_mean_horse('actual_weight'))
  mean_draw = int(get_mean_horse('draw'))
  
  # create the features and label for each horse
  race2Xy = read_Xy(race_details, horse_details, user_feats, word2id, id2word)

  # create splits for training, validation and testing
  test_percent = 0.2
  valid_percent = 0.1
  num_test_records = int(test_percent*len(race_details))
  num_valid_records = int(valid_percent*len(race_details))
  num_train_records = len(race_details) - num_test_records - num_valid_records
  all_races = list(race_details)
  random.shuffle(all_races)
  test_races = all_races[0:num_test_records]
  valid_races = all_races[num_test_records:num_test_records+num_valid_records]
  train_races = all_races[num_test_records+num_valid_records:]  
  
  # create the numpy matrix for features and labels
  train_X, train_y = create_matrix(train_races, race2Xy)
  
  # compute the feature standardizer from training set
  scaler = preprocessing.StandardScaler().fit(train_X)

  # standardize the training features
  train_X = scaler.transform(train_X)
  
  # create the preprocessed training file
  w = open('data/proc/train.txt', 'w')
  w.write("%d,%d\n"%(train_X.shape[0], train_X.shape[1]))
  for ri in range(train_X.shape[0]):
    res = '%d'%train_y[ri]
    for ci in range(train_X.shape[1]):
      res = res + ' %f'%train_X[ri][ci]
    w.write('%s\n'%res)
  w.close()

  # create the preprocessed validation file
  w = open('data/proc/valid.txt', 'w')
  for race in valid_races:
    horses = race2Xy[race]
    w.write("%d\n"%len(horses))
    random.shuffle(horses)
    for horse in horses:
      hid, hfeat, gold = horse
      res = '%d'%gold
      hfeat = scaler.transform(np.array(hfeat).reshape(1, -1))
      for cx in hfeat[0]:
        res = res + ' %f'%cx
      w.write("%s\n"%res)
  w.close()

  # create the preprocessed testing file
  w = open('data/proc/test.txt', 'w')
  for race in test_races:
    horses = race2Xy[race]
    w.write("%d\n"%len(horses))
    random.shuffle(horses)
    for horse in horses:
      hid, hfeat, gold = horse
      res = '%d'%gold
      hfeat = scaler.transform(np.array(hfeat).reshape(1, -1))
      for cx in hfeat[0]:
        res = res + ' %f'%cx
      w.write("%s\n"%res)
  w.close()





