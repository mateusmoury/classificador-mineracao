from os import listdir
from os.path import isfile, join

class CorpusReader:

  def __init__(self):
    self.train = {}
    self.test = {}
    self.topics = ['baseball', 'christian', 'guns']

  def read_data(self, subset, topic):
    directory = './news_groups/' + subset + '/' + topic + '/'
    files = [f for f in listdir(directory) if isfile(join(directory, f)) and f[0] != '.']

    text_from_files = []
    for f in files:
      with open(join(directory, f), 'r', encoding='iso-8859-15') as my_file:
        text_from_files.append(my_file.read())

    if subset is "train":
      self.train[topic] = text_from_files
    else:
      self.test[topic] = text_from_files

  def run(self):
    for topic in self.topics:
      self.read_data('train', topic)
      self.read_data('test', topic)

if __name__ == '__main__':
  x = CorpusReader()
  x.run()
  print(x.train['baseball'][1])
  print(x.test['guns'][2])
