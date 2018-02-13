from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF

import elizabeth


def main(train_x, train_y, test_x, test_y=None, idf=False, base='gs', asm=False):
    # Load : DF[id, url, text, tokens, label?]
    # The DataFrames only have a labels column if labels are given.
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.preprocess.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.preprocess.load(test_x, test_y, base=base, kind=kind)

    prep = elizabeth.preprocess.Preprocessor()
    prep = prep.tf()
    if idf: prep = prep.idf()
    train = prep.fit(train)
    test = prep.transform(test)


    # Naive Bayes : DF[id, url, text, tokens, label?, tf, tfidf, rawPrediction, probability, prediction]
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
