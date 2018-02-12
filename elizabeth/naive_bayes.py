from pyspark.ml.classification import NaiveBayes

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs', asm=False):
    kind = 'asm' if asm else 'bytes'

    train_x = elizabeth.preprocess.load_data(train_x, base=base, kind=kind)
    train_y = elizabeth.preprocess.load_labels(train_y)
    train = train_x.join(train_y, on='id')

    test_x = elizabeth.preprocess.load_data(test_x, base=base, kind=kind)
    test_y = elizabeth.preprocess.load_labels(test_y) if test_y else None
    test = test_x.join(test_y, on='id') if test_y else test_x

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
