import re
from pathlib import Path

import pyspark

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
        base (str):
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
        manifest:
            Path or URL of the manifest file.
        base (str):
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
    id_mapper = lambda id: lambda x: (id, x[0], x[1])
    data = ((id, ctx.wholeTextFiles(url)) for id, url in manifest)  # (id, RDD[url, text])
    data = [rdd.map(id_mapper(id)) for id, rdd in data]             # [RDD[id, url, text]]
    data = ctx.union(data)                                          # RDD[id, url, text]

    # Create a DataFrame.
    data = spark.createDataFrame(data, ['id', 'url', 'text'])
    return data


def load_lines(manifest, base='gs', kind='bytes'):
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Special base paths
    if base == 'https': base = 'https://storage.googleapis.com/uga-dsp/project2/data'
    if base == 'gs': base = 'gs://uga-dsp/project2/data'

    # Read the manifest as an iterator over (id, url).
    # We use Spark to build the iterator to support hdfs etc.
    manifest = str(manifest)  # cast to str to support pathlib.Path etc.
    manifest = ctx.textFile(manifest)  # RDD[hash]
    manifest = manifest.map(hash_to_url(base=base, kind=kind))  # RDD[url]
    manifest = manifest.zipWithIndex()  # RDD[url, id]
    manifest = manifest.map(lambda x: (x[1], x[0]))  # RDD[id, url]
    manifest = manifest.toLocalIterator()

    id_mapper = lambda id: lambda x: (id, x)
    rdds = [ctx.textFile(url)                               # RDD[line]
                .map(lambda line: line.split()[1:])         # RDD[tokens]
                .filter(lambda tokens: '??' not in tokens)  # discard lines with '??' bytes
                .map(id_mapper(id))                         # RDD[id, tokens]
            for id, url in manifest]
    lines = ctx.union(rdds)

    return lines.toDF(['id', 'tokens'])


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
        labels: Path or URL of the label file.

    Returns:
        DataFrame[id: bigint, label: string]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Cast to str to support pathlib.Path etc.
    labels = str(labels)

    # Read the labels as an RDD[id, url].
    labels = ctx.textFile(labels)                     # RDD[label]
    labels = labels.zipWithIndex()                    # RDD[label, id]
    labels = labels.map(lambda x: (x[1], int(x[0])))  # RDD[id, label]

    # Create a DataFrame.
    labels = spark.createDataFrame(labels, ['id', 'label'])
    return labels


@elizabeth.udf([str])
def split_bytes(text):
    '''Splits the text of a bytes file into a list of bytes.
    '''
    bytes = []
    for line in text.split('\n'):
        try:
            (addr, *vals) = line.split()
            bytes += vals
        except ValueError:
            # ValueError occurs on invalid hex,
            # e.g. the missing byte symbol '??'.
            # For now, we discard the whole line. See #6.
            # https://github.com/dsp-uga/elizabeth/issues/6
            continue
    return bytes


@elizabeth.udf([str])
def split_asm(text):
    '''Splits the text of an asm file into a list of opcodes.
    '''
    opcodes = []
    pattern = re.compile(r'\.([a-z]+):([0-9A-F]+)((?:\s[0-9A-F]{2})+)\s+([a-z]+)(?:\s+([^;]*))?')
    for line in text.split('\n'):
        match = pattern.match(line)
        if match:
            opcode = match[4]
            opcodes.append(opcode)
    return opcodes
