import pyspark
from pyspark.ml.linalg import Vectors, SparseVector
from collections import defaultdict


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


def data_type(dtype, not_null=False):
    '''Convert `dtype` to a valid `pyspark.sql.types.DataType`.

    Args:
        dtype:
            Some indicator of the type. It may be
            - an existing `DataType`,
            - a string naming the type,
            - a Python type, like `str`,
            - or a list containing exactly one element,
              interpreted as an `ArrayType` of the given the element type.
        not_null (bool):
            Should values be allowed to be null.
            This is most applicable for `ArrayType`s.

    Returns:
        pyspark.sql.types.DataType
    '''
    types = {
        'null': pyspark.sql.types.NullType(),
        'string': pyspark.sql.types.StringType(),
        'binary': pyspark.sql.types.BinaryType(),
        'boolean': pyspark.sql.types.BooleanType(),
        'date': pyspark.sql.types.DateType(),
        'timestamp': pyspark.sql.types.TimestampType(),
        'double': pyspark.sql.types.DoubleType(),
        'float': pyspark.sql.types.FloatType(),
        'byte': pyspark.sql.types.ByteType(),
        'integer': pyspark.sql.types.IntegerType(),
        'long': pyspark.sql.types.LongType(),
        'short': pyspark.sql.types.ShortType(),

        None: pyspark.sql.types.NullType(),
        str: pyspark.sql.types.StringType(),
        bool: pyspark.sql.types.BooleanType(),
        float: pyspark.sql.types.DoubleType(),
        int: pyspark.sql.types.LongType(),
    }

    # If dtype is a list, interpret it as an ArrayType.
    if isinstance(dtype, list):
        assert len(dtype) == 1
        element_type = data_type(dtype[0])
        dtype = pyspark.sql.types.ArrayType(element_type, containsNull=(not not_null))

    # If dtype is not a DataType, then look it up in the table.
    if not isinstance(dtype, pyspark.sql.types.DataType):
        dtype = types[dtype]

    assert isinstance(dtype, pyspark.sql.types.DataType)
    assert not not_null or dtype != pyspark.sql.types.NullType
    return dtype


def udf(dtype):
    '''A decorator for Spark User Defined Functions (UDF).

    UDFs are used to apply transforms to DataFrames.

    Args:
        dtype:
            The return type of the UDF, interpreted by `elizabeth.data_type`.

    Returns:
        A decorator which converts a function to a `pyspark.sql.functions.udf`.
    '''
    dtype = data_type(dtype)
    return lambda f: pyspark.sql.functions.udf(f, dtype)


def sparse_add(v1, v2):
    # TODO: should this be here?
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    values = defaultdict(float)  # Dictionary with default value 0.0
    # Add values from v1
    for i in range(v1.indices.size):
        values[v1.indices[i]] += v1.values[i]
    # Add values from v2
    for i in range(v2.indices.size):
        values[v2.indices[i]] += v2.values[i]
    return Vectors.sparse(v1.size, dict(values))