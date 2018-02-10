"""
Object-oriented Naive Bayes classifier using MLLib

TODO: add more details
"""

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.regression import LabeledPoint
from collections import defaultdict

from elizabeth import context
from elizabeth import preprocess


def add_sparse(v1, v2):
    """
    Adds two sparse vectors

    Attribution: https://stackoverflow.com/questions/32981875/how-to-add-two-sparse-vectors-in-spark-using-python
    TODO: this should be moved to some other module
    """
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    values = defaultdict(float) # Dictionary with default value 0.0
    # Add values from v1
    for i in range(v1.indices.size):
        values[v1.indices[i]] += v1.values[i]
    # Add values from v2
    for i in range(v2.indices.size):
        values[v2.indices[i]] += v2.values[i]
    return Vectors.sparse(v1.size, dict(values))


class NaiveBayesClassifier:
    def __init__(self, ctx):
        """
        Initializes the classifier using the provided SparkContext
        The provided SparkContext will be used to load data

        :param ctx: The SparkContext to use with this classifier
        """
        self.ctx = ctx
        self.model = None

    def train(self, manifest, labels):
        """ Train the classifier

        The manifest file contains the hashes for the documents used to train this classifier, one per line
        The labels file contains the labels for the training documents referenced in the manifest file

        Successful training of the Classifier should cause self.model to be set

        :param manifest: Path or URL of the manifest file.
        :param labels: Path or URL of the labels file.
        :return: None
        """

        hashingTF = HashingTF(numFeatures=256)  # used to compute term frequencies

        def _line_to_byte_array(line):
            tokens = line.split()[1:]  # throw away the line pointer
            return tokens

        data = preprocess.load_data(ctx=self.ctx, manifest=manifest)            # RDD[id, line]
        labels = preprocess.load_labels(ctx=self.ctx, labels=labels)            # RDD[id, label]
        tokenized_lines = data.mapValues(_line_to_byte_array)
        term_counts = tokenized_lines.mapValues(lambda x: hashingTF.transform(x).toArray())   # RDD[id, term_line_count]
        term_counts = term_counts.reduceByKey(lambda x, y: x+y)                 # RDD[id, term_freq]
        term_counts.cache()

        tf = term_counts.map(lambda x: x[1])                                    # RDD[term_freq]
        idf = IDF().fit(tf)                                                     # IDFModel

        tfidf = term_counts.mapValues(idf.transform)                            # RDD[id, tfidf_vec]

        labeled_data = labels.join(tfidf).map(lambda x: x[1])                   # RDD[label, tfidf_vec]
        labeled_points = labeled_data.map(lambda x: LabeledPoint(x[0], x[1]))   # RDD[LabeledPoint]

        self.model = NaiveBayes.train(labeled_points, lambda_=1.0)

    def classify(self, manifest):
        """ Classify unlabeled data

        The manifest file contains the hashes for the documents to be labelled

        :param manifest:
        :return: RDD[id, predicted_label]
        """
        hashingTF = HashingTF(numFeatures=256)  # used to compute term frequencies

        def _line_to_byte_array(line):
            tokens = line.split()[1:]  # throw away the line pointer
            return tokens

        data = preprocess.load_data(ctx=self.ctx, manifest=manifest)            # RDD[id, line]
        tokenized_lines = data.mapValues(_line_to_byte_array)
        term_counts = tokenized_lines.mapValues(lambda x: hashingTF.transform(x).toArray())   # RDD[id, term_line_count]
        term_counts = term_counts.reduceByKey(lambda x, y: x+y)                 # RDD[id, term_freq]
        term_counts.cache()

        tf = term_counts.map(lambda x: x[1])                                    # RDD[term_freq]
        idf = IDF().fit(tf)                                                     # IDFModel

        tfidf = term_counts.mapValues(idf.transform)                            # RDD[id, tfidf_vec]

        labelled = tfidf.mapValues(lambda x: self.model.predict(x))
        return labelled

    def evaluate(self, manifest, labels):
        """ Evaluate the classifier on a labelled test set

        The manifest file contains the hashes for the documents used for evaluation
        The labels file contains the labels for the testing documents referenced in the manifest file

        :param manifest: Path or URL of the manifest file.
        :param labels: Path or URL of the labels file.
        :return: float expressing percentage of correctly classified examples
        """
        predictions = self.classify(manifest)
        labels = preprocess.load_labels(ctx=self.ctx, labels=labels)
        prediction_and_label = labels.join(predictions).map(lambda x: x[1])

        return prediction_and_label.filter(lambda x: x[0] == x[1]).count() / labels.count()


if __name__ == '__main__':
    # test this classifier
    ctx = context()
    classifier = NaiveBayesClassifier(ctx)
    classifier.train('gs://uga-dsp/project2/files/X_small_train.txt', 'gs://uga-dsp/project2/files/y_small_train.txt')
    classifier.evaluate('gs://uga-dsp/project2/files/X_small_test.txt', 'gs://uga-dsp/project2/files/y_small_test.txt')
