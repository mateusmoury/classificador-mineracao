from math import log

class TfIdf:

  def __init__(self, documents):
    self.documents = documents
    self.document_frequency = {}
    self.word_to_id = {}
    self.tf_idf_vector = []

  def generate_dictionary(self):
    word_count = 0
    for document in self.documents:
      for word in document:
        if word not in self.word_to_id:
          self.word_to_id[word] = word_count
          word_count += 1

    for word, word_id in self.word_to_id.items():
      self.document_frequency[word_id] = sum(1 for document in self.documents if word in document)

  def generate_tf_idf_vector(self):
    for document in self.documents:
      tf_vector = [(self.word_to_id[word], 1 + log(document.count(word))) for word in self.word_to_id if document.count(word) is not 0]
      self.tf_idf_vector.append([(x[0], x[1] * log(len(self.documents) / self.document_frequency[x[0]])) for x in tf_vector])

  def run(self):
    self.generate_dictionary()
    self.generate_tf_idf_vector()

    for i in range(0, len(self.tf_idf_vector)):
      self.tf_idf_vector[i] = list(filter(lambda x: x[1] != 0.0, self.tf_idf_vector[i]))
      self.tf_idf_vector[i].sort()


if __name__ == '__main__':
  x = TfIdf([['oi', 'mateus', 'oi', 'nome', 'mateus', 'oi', 'basquete'], ['mateus', 'basquete'], ['ola']])
  x.run()
  for vector in x.tf_idf_vector:
    print(vector)


