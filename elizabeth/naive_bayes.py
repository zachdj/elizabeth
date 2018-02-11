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

    def fit(self, x, y, data_col='bytes'):
        '''Fit the model to some data.

        The data (x) and labels (y) are joind on the column `id`.

        Args:
            x (DataFrame):
                The training data. It must contain a column `id` that uniquely
                identifies each instance, and a data column of `ArrayType(str)`
                that provides tokens.
            y (DataFrame):
                The training labels. It must contain a column `id` that uniquely
                identifies each instance, and a column `label` giving the label
                of each instance.
            data_col (str):
                The name of the data column.

        Returns:
            self
        '''
        tf = pyspark.ml.feature.CountVectorizer(inputCol=data_col, outputCol='tf').fit(x)
        x = tf.transform(x)

        idf = pyspark.ml.feature.IDF(inputCol='tf', outputCol='tfidf').fit(x)
        x = idf.transform(x)

        x = x.join(y, on='id')
        nb = pyspark.ml.classification.NaiveBayes(featuresCol='tfidf').fit(x)

        self._tf = tf
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
            A new DataFrame like x with new columns.
            TODO: what are the name(s) of the new columns?
        '''
        x = self._tf.transform(x)
        x = self._idf.transform(x)
        return self._model.transform(x)
