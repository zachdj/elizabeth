from pyspark.ml.classification import NaiveBayes

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs', asm=False):
    kind = 'asm' if asm else 'bytes'

    train = elizabeth.preprocess.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.preprocess.load(test_x, test_y, base=base, kind=kind)
    nb = NaiveBayes(featuresCol='tfidf', labelCol='label').fit(train)

    if test_y:
        # If test_y is given, we print out a score rather than a prediction.
        raise NotImplementedError()
    else:
        # TODO: Print the output _exactly_ as expected by AutoLab.
        test = nb.transform(test)
        test = test.drop('text', 'tokens', 'tf', 'tfidf')
        h = test.orderBy('id').collect()
        print(*h, sep='\n')
