from . import preprocess

import pyspark


def context(**kwargs):
    conf = pyspark.SparkConf()
    conf.setAppName('elizabeth')
    conf.setAll(kwargs.items())
    ctx = pyspark.SparkContext()
    return ctx


def session(**kwargs):
    conf = pyspark.SparkConf()
    conf.setAppName('elizabeth')
    conf.setAll(kwargs.items())
    sess = (pyspark.sql.SparkSession
        .builder
        .config(conf=conf)
        .getOrCreate())
    return sess
