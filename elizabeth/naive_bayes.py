from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs', asm=False):
    # Load : DF[id, url, text, tokens, labels?]
    # The DataFrames only have a labels column if labels are given.
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.preprocess.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.preprocess.load(test_x, test_y, base=base, kind=kind)

    # TF-IDF : DF[id, url, text, tokens, labels?, tf, tfidf]
    tf = CountVectorizer(inputCol='tokens', outputCol='tf').fit(train)
    train, test = tf.transform(train), tf.transform(test)
    idf = IDF(inputCol='tf', outputCol='tfidf').fit(train)
    train, test = idf.transform(train), idf.transform(test)

    # Naive Bayes : DF[id, url, text, tokens, labels?, tf, tfidf, rawPrediction, probability, prediction]
    nb = NaiveBayes(featuresCol='tfidf', labelCol='label').fit(train)
    test = nb.transform(test)

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
