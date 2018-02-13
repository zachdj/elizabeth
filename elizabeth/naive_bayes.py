from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF, NGram

import elizabeth


def main(train_x, train_y, test_x, test_y=None, idf=False, ngram=1, base='gs', asm=False):
    # Load : DF[id, url, text, features, label?]
    # The DataFrames only have a labels column if labels are given.
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.load(test_x, test_y, base=base, kind=kind)

    # Train the preprocessor and transform the data.
    prep = elizabeth.Preprocessor()
    prep.add(NGram(n=ngram))
    prep.add(CountVectorizer())
    if idf: prep.idf(IDF())
    train = prep.fit(train)
    test = prep.transform(test)

    # Naive Bayes : DF[id, url, text, features, label?, rawPrediction, probability, prediction]
    nb = NaiveBayes().fit(train)
    test = nb.transform(test)
    test = test.withColumn('prediction', test.prediction + 1)

    # If labels are given for the test set, print a score.
    if test_y:
        test = test.orderBy(test.id)
        test = test.withColumn('correct', (test.label == test.prediction).cast('double'))
        test = test.select(avg(test.correct))
        print(test.show())

    # If no labels are given for the test set, print predictions.
    else:
        test = test.orderBy(test.id).select(test.prediction)
        test = test.rdd.map(lambda row: int(row.prediction))
        test = test.toLocalIterator()
        print(*test, sep='\n')
