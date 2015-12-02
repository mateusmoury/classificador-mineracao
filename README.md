
# classificador-mineracao
Classificador de texto implementado como projeto final de Mineração da Web - CIn 2015.2

## usando o ambiente virtual

O ambiente virtual serve para não termos problemas em rodar o projeto em diferentes máquinas que estão o utilizando. O ambiente virtual nada mais é do que a criação de um ambiente onde todas as bibliotecas serão instaladas a parte no seu computador. Isso é, as bibliotecas que são adicionadas lá não são adicionadas ao seu computador, e não podemos acessá-las se não estivermos no ambiente virtual. Para ligar o ambiente virtual, rodar o comando:
$ source venv/bin/activate
Feito isso, rodar o comando
$ pip install -r requirements.txt
Para obter todas as bibliotecas e baixá-las para o ambiente virtual da sua máquina. 
Caso alguma biblitoeca nova seja instalada no projeto, então rodar o comando
$ pip freeze > requirements.txt

## usando corpus_reader.py

O corpus reader nada mais faz do que ler os documentos do nosso corpus. Depois de cria e rodar uma instância de CorpusReader.run(), podemos acessar os documentos em dois mapas: train e test, onde estao os documentos de treinamento e teste respectivamente. Para acessar por topicos, basta escolher entre baseball, christian e guns. train["guns"] vai retornar um array de strings, onde cada string é um texto de treinamento do topico guns.

## usando o tf_idf.py

O construtor de uma instância TfIdf recebe um array de documentos, onde cada documento é uma string. Ele vai retornar um vetor de vetores. Cada vetor retornado corresponde a um documento de entrada. Cada vetor é composto por uma lista de tuplas (x, y), onde x é o ID da palavra e y é o TF-IDF daquela palavra no documento. O tf foi calculado usando o método: 1 + log(frequencia_da_palavra_no_documento), e o idf usando log(numero_de_documentos/quantidade_de_documentos_que_palavra_aparece). 

## usando o preprocess.py

Após a criação de uma instância `preprocess`, o método `process` recebe uma lista L de strings e vai retornar um lista de tamanho L de lista de strings, onde cada lista interior é a sentença após realizado o processamento. As etapas de processamento realizadas foram:

- Tokenização
- Stemming (algoritmo de Porter)
- Remoção de stopwords (coleção de 127 palavras de stopwords do NLTK), incluindo dígitos e pontuação
