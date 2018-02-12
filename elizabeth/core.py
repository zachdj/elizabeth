import pyspark


def session(**kwargs):
    '''Get or create the global `SparkSession`.

    Kwargs:
        Forwarded to `SparkConf.setAll` to initialize the session.

    Returns:
        SparkSession
    '''
    conf = pyspark.SparkConf()
    conf.setAppName('elizabeth')
    conf.setAll(kwargs.items())
    sess = (pyspark.sql.SparkSession
        .builder
        .config(conf=conf)
        .getOrCreate())
    return sess
