import pyspark
import pyspark.ml.feature
import pyspark.ml.classification

import elizabeth


class NaiveBayes:
    '''A NaiveBayes model with TF-IDF.
    '''

    def __init__(self):
        '''Initialize the model.
        '''
        self._tf = None
        self._idf = None
        self._model = None

    def fit(self, x, y, data_col='tokens'):
        '''Fit the model to some data.

        The data (x) and labels (y) are joined on the column `id`.

        Args:
            x (DataFrame):
                The training data.  It must contain a data column 'tokens'
                that provides tokens from a line of text in the training set,
                and a column 'id' that uniquely identifies the document
                that contains that line of text.
            y (DataFrame):
                The training labels. It must contain a column `id` that uniquely
                identifies each instance, and a column `label` giving the label
                of each instance.
            data_col (str):
                The name of the data column.

        Returns:
            self
        '''

        hashingTF = pyspark.ml.feature.HashingTF(numFeatures=512, inputCol=data_col, outputCol='tf')
        tf = hashingTF.transform(x).drop('tokens')    # computes term frequencies for each vector of tokens
        # sum up term frequencies for each document
        tf_summed = tf.rdd \
            .map(lambda row: (row.id, row.tf)) \
            .reduceByKey(lambda x, y: elizabeth.sparse_add(x, y))\
            .toDF(['id', 'tf'])
        idf = pyspark.ml.feature.IDF(inputCol='tf', outputCol='tfidf').fit(tf_summed)
        tfidf = idf.transform(tf_summed).drop('tf')

        labelled = tfidf.join(y, on='id')
        nb = pyspark.ml.classification.NaiveBayes(featuresCol='tfidf').fit(labelled)

        self._tf = hashingTF
        self._idf = idf
        self._model = nb
        return self

    def transform(self, x):
        '''Transforms a dataset by adding a column of predicted labels.

        Args:
            x (DataFrame):
                The data to transform. It must contain a data column of
                `ArrayType(str)` that provides tokens.

        Returns:
            A new DataFrame like x with new columns 'prediction' containing the
            predicted class label and 'probability' containing a probability
            vector for all class labels
        '''
        # TODO: this isn't very DRY
        x = self._tf.transform(x).drop('tokens')
        x = x.rdd \
            .map(lambda row: (row.id, row.tf)) \
            .reduceByKey(lambda x, y: elizabeth.sparse_add(x, y)) \
            .toDF(['id', 'tf'])
        x = self._idf.transform(x)
        return self._model.transform(x)


def main(train_x, train_y, test_x, test_y=None, base='gs', asm=False):
    nb = elizabeth.naive_bayes.NaiveBayes()

    if asm:
        train_x = elizabeth.preprocess.load_data(train_x, base=base, kind='asm')
        train_x = train_x.withColumn('tokens', elizabeth.preprocess.split_asm(train_x.text))
        test_x = elizabeth.preprocess.load_data(test_x, base=base, kind='asm')
        test_x = test_x.withColumn('tokens', elizabeth.preprocess.split_asm(test_x.text))
    else:
        train_x = elizabeth.preprocess.load_lines(train_x, base=base, kind='bytes')
        test_x = elizabeth.preprocess.load_lines(test_x, base=base, kind='bytes')

    train_y = elizabeth.preprocess.load_labels(train_y)
    test_y = elizabeth.preprocess.load_labels(test_y) if test_y else None

    nb.fit(train_x, train_y)

    if test_y:
        # If test_y is given, we print out a score rather than a prediction.
        # TODO: We currently print a prediction since the scoring code hasn't been written.
        predictions = nb.transform(test_x)
        predictions.show()
    else:
        # TODO: Print the output _exactly_ as expected by AutoLab.
        predictions = nb.transform(test_x)
        predictions.show()
