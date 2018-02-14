"""
    Random Forest classifier

    Features:
        - frequencies of the top 500 bigrams from the binary files
        - frequencies of the top 1000 4-grams from the binary files
"""

from pyspark.sql.functions import avg
from pyspark.ml.classification import RandomForestClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, CountVectorizer, NGram, StringIndexer, IndexToString

import elizabeth


def main(train_x, train_y, test_x, test_y=None, base='gs'):
    # Load : DF[id, url, features, label?]
    # The DataFrames only have a labels column if labels are given.
    train = elizabeth.load(train_x, train_y, base=base, kind='bytes').drop('text')
    test = elizabeth.load(test_x, test_y, base=base, kind='bytes').drop('text')

    # convert the string labels to numeric indices
    # the handleInvalid param allows the label indexer to deal with labels that weren't seen during fitting
    label_indexer = StringIndexer(inputCol='label', outputCol='indexedLabel', handleInvalid="skip")
    label_indexer = label_indexer.fit(train)
    train = label_indexer.transform(train)
    # the test set won't always have labels
    if test_y is not None:
        test = label_indexer.transform(test)

    index_labeller = IndexToString(inputCol='prediction', outputCol='predictedClass', labels=label_indexer.labels)

    # transform the data
    # create ngrams
    two_gram = NGram(n=2, inputCol='features', outputCol='twoGrams')
    four_gram = NGram(n=4, inputCol='features', outputCol='fourGrams')
    train = two_gram.transform(train)
    train = four_gram.transform(train)
    test = two_gram.transform(test)
    test = four_gram.transform(test)

    # create ngram frequencies
    cv = CountVectorizer(inputCol="twoGrams", outputCol="twoGramCounts", vocabSize=500)
    cv_model = cv.fit(train)
    train = cv_model.transform(train).drop('twoGrams')
    test = cv_model.transform(test).drop('twoGrams')

    cv = CountVectorizer(inputCol="fourGrams", outputCol="fourGramCounts", vocabSize=1000)
    cv_model = cv.fit(train)
    train = cv_model.transform(train).drop('fourGrams')
    test = cv_model.transform(test).drop('fourGrams')

    # drop the tokens loaded by the preprocessor
    train = train.drop('features')
    test = test.drop('features')

    # combine features into a single feature vector, and drop the individual columns
    assembler = VectorAssembler(inputCols=['twoGramCounts', 'fourGramCounts'], outputCol='features')
    train = assembler.transform(train)\
        .drop('twoGramCounts', 'fourGramCounts')
    test = assembler.transform(test)\
        .drop('twoGramCounts', 'fourGramCounts')

    # instantiate a binary GBT as the base classifier for One-vs-Many
    rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol='features',
                                 numTrees=20, maxDepth=10, minInfoGain=0.0)
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
