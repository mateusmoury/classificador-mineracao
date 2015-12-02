from corpus_reader import CorpusReader
from preprocess import PreProcess
from tf_idf import TfIdf

if __name__ == '__main__':
  reader = CorpusReader()
  reader.run()
  parser = PreProcess()
  parsed_trainning_documents = {}
  for k, v in reader.train.items():
    parsed_trainning_documents[k] = parser.process(v)
  # Entrada para o tf-idf, devemos anotar os documentos com suas classes.
  # Receberá como entrada um array de tuplas: ([tokens], classe)
  parsed_trainning_documents_with_classes = []
  for k in parsed_trainning_documents.keys():
    parsed_trainning_documents_with_classes += [(v, k) for v in parsed_trainning_documents[k]]
  # Execucao tf-idf
  tf_idf_calculator = TfIdf(parsed_trainning_documents_with_classes)
  tf_idf_calculator.run()
  for vector in tf_idf_calculator.results:
    print(vector[0])
    print(vector[1])
