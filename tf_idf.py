from math import log

class TfIdf:

  def __init__(self, annotated_documents):
    self.documents = [x[0] for x in annotated_documents]
    self.classes = [x[1] for x in annotated_documents]
    self.document_frequency = {}
    self.word_to_id = {}
    self.results = []

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
    tf_idf_vector = []
    for document in self.documents:
      tf_vector = [(self.word_to_id[word], 1 + log(document.count(word))) for word in self.word_to_id if document.count(word) is not 0]
      tf_idf_vector.append([(x[0], x[1] * log(len(self.documents) / self.document_frequency[x[0]])) for x in tf_vector])

    return tf_idf_vector

  def run(self):
    self.generate_dictionary()
    tf_idf_vector = self.generate_tf_idf_vector()

    for i in range(0, len(tf_idf_vector)):
      tf_idf_vector[i] = list(filter(lambda x: x[1] != 0.0, tf_idf_vector[i]))
      tf_idf_vector[i].sort()

    for i in range(0, len(tf_idf_vector)):
      self.results += [(tf_idf_vector[i], self.classes[i])]


if __name__ == '__main__':
  x = TfIdf([(['oi', 'mateus', 'oi', 'nome', 'mateus', 'oi', 'basquete'], 'classe1'), (['mateus', 'basquete', 'ola'], 'classe2')])
  x.run()
  for vector in x.results:
    print(vector[0])
    print(vector[1])


