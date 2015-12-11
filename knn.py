from math import sqrt

class KNN:
  def __init__(self, docs, kk=1, met='cosine'):
    self.topics = ['baseball', 'christian', 'guns']
    self.k = min(kk, len(docs))
    self.documents = docs
    self.metric = met
    
    if self.metric == 'cosine':
      for (a, b) in self.documents:
        self.normalize(a)
  
  def normalize(self, vector):
    v_len = sqrt(sum([pair[1]*pair[1] for pair in vector]))
    for i in range(len(vector)):
      pair = vector[i]
      vector[i] = (pair[0], pair[1]/v_len)
  
  def score(self, doc, query):
    score = 0
    
    for (k, v) in doc:
      if k in query:
        if self.metric == 'cosine':
          score += v*query[k]
        else:
          dv = v - query[k]
          score += dv*dv
    
    # uso distância euclidiana quadrada mesmo, não faz diferença
    if self.metric == 'euclid':
      return -score
    else:
      return score
  
  def classify(self, query):
    # query é um vetor no mesmo formato que os de treinamento
    if self.metric == 'cosine':
      self.normalize(query)
    query = {a:b for (a, b) in query}
    
    # calcula os scores pra cada documento de treinamento
    scores = [(self.score(document[0], query), document[1]) for document in self.documents]
    scores.sort()
    scores.reverse()
    
    # conta as classes dos k primeiros
    topic_count = {t:0 for t in self.topics}
    for i in range(self.k):
      topic_count[scores[i][1]] += 1
    
    # e retorna o mais frequente
    topic_count_r = [(b, a) for (a, b) in topic_count.items()]
    topic_count_r.sort()
    return topic_count_r[-1][1]

if __name__ == '__main__':
  vector = [(0, 1.0), (1, 1.0)]
  knn = KNN([])
  knn.normalize(vector)
  print(vector)

