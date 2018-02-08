import pyspark


def context(**kwargs):
    conf = pyspark.SparkConf()
    conf.setAppName('elizabeth')
    conf.setAll(kwargs.items())
    ctx = pyspark.SparkContext()
    return ctx
