from corpus_reader import CorpusReader
from preprocess import PreProcess
from tf_idf import TfIdf
from knn import KNN
from metrics import MetricsGenerator
from pprint import pprint as pp

if __name__ == '__main__':
  print('reading...')
  reader = CorpusReader()
  reader.run()
  
  parser = PreProcess()
  parsed_trainning_documents = {}
  print('processing...')
  for k, v in reader.train.items():
    parsed_trainning_documents[k] = parser.process(v)
  
  # Entrada para o tf-idf, devemos anotar os documentos com suas classes.
  # Receberá como entrada um array de tuplas: ([tokens], classe)
  parsed_trainning_documents_with_classes = []
  for k in parsed_trainning_documents.keys():
    parsed_trainning_documents_with_classes += [(v, k) for v in parsed_trainning_documents[k]]
  
  # Execução tf-idf
  print('generating tf.idf...')
  tf_idf_calculator = TfIdf(parsed_trainning_documents_with_classes)
  tf_idf_calculator.run()
  
  # testa os parâmetros do knn: métrica de distância e valor de K
  for metric in ['cosine', 'euclid']:
    for k in range(5, 11, 2):
      knn = KNN(tf_idf_calculator.results, k, metric)
    
      # confusion_matrix[A][B] = quantas vezes um documento da classe A foi atribuído à classe B
      topics = ['baseball', 'christian', 'guns']
      confusion_matrix = {topic:{t:0 for t in topics} for topic in topics}
      
      print_log = False
      i = 0
      ytrue = []
      ypred = []
      for topic in topics:
        for doc in reader.test[topic]:
          ytrue.append(topic)
          # classifica os documentos de teste
          words = parser.process_sent(doc)
          query = tf_idf_calculator.generate_doc_vector(words)
          result = knn.classify(query)
          confusion_matrix[topic][result] += 1
          ypred.append(result)
          i += 1
          if print_log:
            print('')
            print(i)
            print(doc)
            print(words)
            print(query)
            print(result)
      
      # e imprime os resultados
      print('#'*40)
      s = '#'*10 + (' K=%d || dist=%s ' % (k, metric)) + '#'*10
      print(s)
      print('#'*40)
      mg = MetricsGenerator(ytrue, ypred)
      print('Precision: %.2f' % mg.macro_precision())
      print('Recall: %.2f' % mg.macro_recall())
      print('F1-score: %.2f' % mg.macro_fscore())
      print('Accuracy: %.2f' % mg.accuracy())
      print('---\nConfusion Matrix\n---')
      pp(confusion_matrix)
      print('#'*40)