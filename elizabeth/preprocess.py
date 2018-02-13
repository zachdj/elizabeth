import re
from copy import copy
from pathlib import Path

import pyspark
from pyspark.ml.feature import RegexTokenizer, NGram, CountVectorizer, IDF

import elizabeth


def hash_to_url(hash=None, base='./data', kind='bytes'):
    '''Returns a function mapping document hashes to Google Storage URLs.

    The API for this function is fancy. It can be used directly as a closure:

        >>> my_rdd.map(hash_to_url)

    Or it can be used to return a closure with different defaults:

        >>> my_rdd.map(hash_to_url(base='https', kind='asm'))

    The first form is convenient for interactive use, but may be confusing or
    unclear for production use. Prefer the second form in that case.

    Args:
        hash (str):
            The hash identifying a document instance.
        base (path):
            The base of the URL or path to the data.
            The data must live at '{base}/{kind}/{hash}.{kind}'
        kind (str):
            The kind of file to use, either 'bytes' or 'asm'.

    Returns:
        If `hash` is given, returns the URL to the document.
        If `hash` is None, returns a closure that transforms a hash to a URL.
    '''
    base = str(base)

    # If base is not a URL, turn it into a `file:` URL.
    # Note that Spark uses `file:`, but Pathlib uses `file://`,
    # so we can't just use `Path.to_uri`
    url = re.compile(r'^[a-zA-Z0-9]+:')
    if not url.match(base):
        base = Path(base).resolve()
        base = f'file:{base}'

    closure = lambda hash: f'{base}/{kind}/{hash}.{kind}'

    if hash is None:
        return closure
    else:
        return closure(hash)


def load_data(manifest, base='gs', kind='bytes'):
    '''Load data from a manifest file into a DataFrame.

    A manifest file gives the hash identifying each document on separate lines.

    The returned DataFrame has columns `id`, `url`, and `text` where `id`
    is a document identifier, `url` is the path to the document, and `text`
    is the contents.

    Note that the document ID is _not_ the same as the hash. The ID is
    guaranteed to uniquely identify one document and the order of the IDs is
    guaranteed to match the order given in the manifest file.

    Args:
        manifest (path):
            Path or URL of the manifest file.
        base (path):
            The base of the URL or path to the data. The special strings 'gs'
            and 'https' expand to the URLs used by Data Science Practicum at
            UGA over the Google Storage and HTTPS protocols respectivly.
        kind (str):
            The kind of file to use, either 'bytes' or 'asm'.

    Returns:
        DataFrame[id: bigint, url: string, text: string]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Special base paths
    if base == 'https': base = 'https://storage.googleapis.com/uga-dsp/project2/data'
    if base == 'gs': base = 'gs://uga-dsp/project2/data'

    # Read the manifest as an iterator over (id, url).
    # We use Spark to build the iterator to support hdfs etc.
    manifest = str(manifest)  # cast to str to support pathlib.Path etc.
    manifest = ctx.textFile(manifest)                           # RDD[hash]
    manifest = manifest.map(hash_to_url(base=base, kind=kind))  # RDD[url]
    manifest = manifest.zipWithIndex()                          # RDD[url, id]
    manifest = manifest.map(lambda x: (x[1], x[0]))             # RDD[id, url]
    manifest = manifest.toLocalIterator()                       # (id, url)

    # Load all files in the base directoy, then join out the ones in the manifest.
    prepend = lambda *args: lambda x: (*args, *x)
    data = ((id, ctx.wholeTextFiles(url)) for id, url in manifest)  # (id, RDD[url, text])
    data = [rdd.map(prepend(id)) for id, rdd in data]               # [RDD[id, url, text]]
    data = ctx.union(data)                                          # RDD[id, url, text]
    data = data.toDF(['id', 'url', 'text'])                         # DF[id, url, text]

    # Tokenization : DF[id, url, text, tokens]
    tokenizer = RegexTokenizer(inputCol='text', outputCol='features', gaps=False)
    if kind == 'bytes': tokenizer.setPattern('(?<= )[0-9A-F]{2}')
    elif kind == 'asm': tokenizer.setPattern('(?<=\.([a-z]+):([0-9A-F]+)((?:\s[0-9A-F]{2})+)\s+)([a-z]+)')
    data = tokenizer.transform(data)
    data = data.drop('text')

    return data.persist()


def load_labels(labels):
    '''Load labels from a label file into a DataFrame.

    A label file corresponds to a manifest file and each line gives the label
    of the document on the same line of the manifest file.

    The returned DataFrame has columns `id` and `label` where `id` is a
    document identifier and `label` is a label for the document.

    Note that the document ID is _not_ the same as the hash. The ID of a
    document is guaranteed to match the ID returned by `load_data` from the
    corresponding manifest file.

    Args:
        labels (path):
            Path or URL of the label file.

    Returns:
        DataFrame[id: bigint, label: string]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Cast to str to support pathlib.Path etc.
    labels = str(labels)

    # Read the labels as a DataFrame[id, url].
    labels = ctx.textFile(labels)                     # RDD[label]
    labels = labels.zipWithIndex()                    # RDD[label, id]
    labels = labels.map(lambda x: (x[1], int(x[0])))  # RDD[id, label]
    labels = labels.toDF(['id', 'label'])             # DF[id, label]

    return labels.persist()


def load(manifest, labels=None, base='gs', kind='bytes'):
    '''Load data and labels into a single DataFrame.

    Labels need not be given. In that case, the result will not have a
    label column.

    Args:
        manifest (path):
            Path to the manifest file for the data.
        labels (path):
            Path to the label file for the data.
        base (path):
            The base path to the data files. The special strings 'gs' and
            'https' expand to the URLs used by Data Science Practicum at UGA
            over the Google Storage and HTTPS protocols respectivly.
        kind (str):
            The kind of file to use, either 'bytes' or 'asm'.

    Returns:
        DataFrame[id: bigint, url: string, text: string, label: string]
    '''
    if labels:
        x = load_data(manifest, base, kind).unpersist()
        y = load_labels(labels).unpersist()
        return x.join(y, on='id').persist()

    else:
        return load_data(manifest, base, kind)


class Preprocessor:
    '''A preprocess pipeline.

    This class is similar to `pyspark.ml.Pipeline`. The benefit is that it
    gives us a central place to define and reuse all of our preprocess steps.

    The class builds a linked-list of preprocess stages. Each stage is defined
    by an Estimator which fits a transformer. During `fit`, the pipeline
    starts at the root, and for each stage it creates a Transformer for that
    stage by fitting the Estimator and transforms the data before passing it to
    the next stage. Likewise diring `transform`, the pipeline starts at the
    root and iterativly transforms the data.

    A stage may be defined by a Transformer directly instead of an Estimator.
    In that case, the fit for that stage is a noop.

    Note when adding new stages, the input column must be 'features' and the
    output column must be 'transform'. The final 'transform' column replaces
    the 'features' column of the input, and maintains the name 'features'.
    '''
    def __init__(self):
        '''Initialize a Preprocessor'''
        self._extended = False  # Can't extend a preprocessor twice
        self._prev = None  # Preprocessor stages are aranged in a linked-list.
        self._estimator = None  # Estimator for this stage, has a `fit` method.
        self._model = None  # Model for this stage, has a `transform` method.

    @property
    def is_root(self):
        '''True if this Preprocessor is the root of the pipeline.'''
        return self._prev is None

    def fit(self, x):
        '''Fit the estimators in this pipeline.

        Args:
            x (DataFrame):
                The dataframe to transform. The column named 'features' will
                be transformed in-place to the new form.

        Returns:
            x transformed by this pipeline.
        '''
        if self.is_root: return x
        x = self._prev.fit(x)
        if self._estimator is not None:
            self._model = self._estimator.fit(x)
        return self._transform(x)

    def transform(self, x):
        '''Transform a dataframe.

        Args:
            x (DataFrame):
                The dataframe to transform. The column named 'features' will
                be transformed in-place to the new form.

        Returns:
            x transformed by this pipeline.
        '''
        if self.is_root: return x
        x = self._prev.transform(x)
        return self._transform(x)

    def _transform(self, x):
        '''Apllies the transformation for this stage in-place.'''
        m = self._model
        x = m.transform(x)
        x = x.drop('features')
        x = x.withColumnRenamed('transform', 'features')
        return x

    def add(self, stage):
        '''Add a new stage to the pipeline.

        Args:
            stage (Estimator or Transformer):
                The new stage.
        '''
        assert self._extended is False
        prev = copy(self)
        self._prev = prev
        if hasattr(stage, 'fit'):
            self._estimator = stage
            self._model = None
        else:
            self._estimator = None
            self._model = stage
        self._extended = True
        return self

    def ngram(self, n, **kwargs):
        '''Add an n-grammer to the pipeline.
        '''
        n = int(n)
        assert n > 0
        if n == 1: return self
        kwargs['inputCol'] = 'features'
        kwargs['outputCol'] = 'transform'
        ngram = NGram(n=n, **kwargs)
        return self.add(model=ngram)

    def tf(self, **kwargs):
        '''Add a count vectorizer to the pipeline.
        '''
        kwargs['inputCol'] = 'features'
        kwargs['outputCol'] = 'transform'
        tf = CountVectorizer(**kwargs)
        return self.add(estimator=tf)

    def idf(self, **kwargs):
        '''Add an IDF estimator to the pipeline.
        '''
        kwargs['inputCol'] = 'features'
        kwargs['outputCol'] = 'transform'
        idf = IDF(**kwargs)
        return self.add(estimator=idf)
