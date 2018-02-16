"""
    Random Forest classifier

    Features:
        - lengths of different sections (HEADER, .text, .idata, .rsrc, etc) of asm file
        - frequency of opcodes
        - frequencies of the top 30 bigrams from the binary files
"""

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, CountVectorizer, NGram, StringIndexer, IndexToString, HashingTF

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs'):
    # generate feature set for asm files
    train_asm_features = elizabeth.preprocess.load(train_x, train_y, base=base,  kind='sect_ops').drop('url')
    test_asm_features = elizabeth.preprocess.load(test_x, test_y, base=base,  kind='sect_ops').drop('url')

    asm_token_counter = CountVectorizer(inputCol='features', outputCol='asm_features', minDF=10).fit(train_asm_features)
    train = asm_token_counter.transform(train_asm_features).drop('features')
    test = asm_token_counter.transform(test_asm_features).drop('features')

    # convert the string labels to numeric indices
    # the handleInvalid param allows the label indexer to deal with labels that weren't seen during fitting
    label_indexer = StringIndexer(inputCol='label', outputCol='indexedLabel', handleInvalid="skip")
    label_indexer = label_indexer.fit(train)
    train = label_indexer.transform(train)
    # the test set won't always have labels
    if test_y is not None:
        test = label_indexer.transform(test)

    index_labeller = IndexToString(inputCol='prediction', outputCol='predictedClass', labels=label_indexer.labels)

    # create and train a Random Forest classifier
    rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol='asm_features',
                                 numTrees=20, maxDepth=10, minInfoGain=0.0, seed=12345)
    model = rf.fit(train)
    prediction = model.transform(test)
    prediction = index_labeller.transform(prediction)  # DF[id, url, ... prediction, predictedClass]

    # If labels are given for the test set, print a score.s
    if test_y:
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(prediction)
        print("\n\tAccuracy on test set: %0.6f\n" % accuracy)

    # If no labels are given for the test set, print predictions.
    else:
        prediction = prediction.orderBy(prediction.id).select(prediction.predictedClass)
        prediction = prediction.rdd.map(lambda prediction: int(prediction.predictedClass))
        prediction = prediction.toLocalIterator()
        print(*prediction, sep='\n')
