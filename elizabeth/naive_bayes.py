from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs', asm=False):
    kind = 'asm' if asm else 'bytes'

    train = elizabeth.preprocess.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.preprocess.load(test_x, test_y, base=base, kind=kind)
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
