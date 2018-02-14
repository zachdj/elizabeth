from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF, NGram, StringIndexer, IndexToString

import elizabeth


def main(train_x, train_y, test_x, test_y=None, idf=False, ngram=1, base='gs', asm=False):
    # Load : DF[id, url, features, label?]
    # The DataFrames only have a labels column if labels are given.
    # We drop the text, since Naive Bayes doesn't use it and we already have all the tokens
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.load(train_x, train_y, base=base, kind=kind).drop('text')
    test = elizabeth.load(test_x, test_y, base=base, kind=kind).drop('text')

    # convert the string labels to numeric indices
    # the handleInvalid param allows the label indexer to deal with labels that weren't seen during fitting
    label_indexer = StringIndexer(inputCol='label', outputCol='indexedLabel', handleInvalid="skip")
    label_indexer = label_indexer.fit(train)
    train = label_indexer.transform(train)
    # the test set won't always have labels
    if test_y is not None:
        test = label_indexer.transform(test)

    index_labeller = IndexToString(inputCol='prediction', outputCol='predictedClass', labels=label_indexer.labels)

    # Train the preprocessor and transform the data.
    prep = elizabeth.Preprocessor()
    prep.add(NGram(n=int(ngram)))
    prep.add(CountVectorizer())
    if idf: prep.add(IDF())
    train = prep.fit(train)
    test = prep.transform(test)

    # Naive Bayes : DF[id, url, text, features, label?, rawPrediction, probability, prediction]
    nb = NaiveBayes(labelCol='indexedLabel').fit(train)
    test = nb.transform(test)
    test = index_labeller.transform(test)  # DF[id, url, ... prediction, predictedClass]

    # If labels are given for the test set, print a score.s
    if test_y:
        test = test.orderBy(test.id)
        test = test.withColumn('correct', (test.label == test.predictedClass).cast('double'))
        test = test.select(avg(test.correct))
        print(test.show())

    # If no labels are given for the test set, print predictions.
    else:
        test = test.orderBy(test.id).select(test.predictedClass)
        test = test.rdd.map(lambda row: int(row.predictedClass))
        test = test.toLocalIterator()
        print(*test, sep='\n')
