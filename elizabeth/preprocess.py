import re
from pathlib import Path

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
    '''Load data from a manifest file into an RDD.

    A manifest file gives the hash identifying each document on separate lines.

    The returned RDD is of the form `RDD[id, line]` where `id` uniquely
    identifies the document and `line` is a line of the document as a string.

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
        RDD[id, line]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Special base paths
    if base == 'https': base = 'https://storage.googleapis.com/uga-dsp/project2/data'
    if base == 'gs': base = 'gs://uga-dsp/project2/data'

    # Read the manifest as an RDD[url, id].
    manifest = str(manifest)  # cast to str to support pathlib.Path etc.
    manifest = ctx.textFile(manifest)                           # RDD[hash]
    manifest = manifest.map(hash_to_url(base=base, kind=kind))  # RDD[url]
    manifest = manifest.zipWithIndex()                          # RDD[url, id]

    # Load all files in the base directoy, then join out the ones in the manifest.
    data = ctx.wholeTextFiles(f'{base}/{kind}')          # RDD[url, text]
    data = manifest.join(data)                           # RDD[url, (id, text)]
    data = data.map(lambda x: (x[1][0], x[0], x[1][1]))  # RDD[id, url, text]

    # Create a DataFrame.
    data = spark.createDataFrame(data, ['id', 'url', 'text'])
    return data


def load_labels(labels):
    '''Load labels from a label file into an RDD.

    A label file corresponds to a manifest file and each line gives the label
    of the document on the same line of the manifest file.

    The returned RDD is of the form `RDD[id, label]` where `id` uniquely
    identifies the document and `label` is a label for the document.

    Note that the document ID is _not_ the same as the hash. The ID of a
    document is guaranteed to match the ID returned by `load_data` from the
    corresponding manifest file.

    Args:
        labels: Path or URL of the label file.

    Returns:
        RDD[id, label]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Cast to str to support pathlib.Path etc.
    labels = str(labels)

    # Read the labels as an RDD[id, url].
    labels = ctx.textFile(labels)                # RDD[label]
    labels = labels.zipWithIndex()               # RDD[label, id]
    labels = labels.map(lambda x: (x[1], x[0]))  # RDD[id, label]

    # Create a DataFrame.
    labels = spark.createDataFrame(labels, ['id', 'label'])
    return labels


def split_bytes(data, no_addr=False):
    '''Splits RDDs of the form `RDD[id, line]` into `RDD[id, (addr, byte)]`
    where `addr` is the address of the byte, and `byte` is the value.

    The input is expected to be loaded from bytes files.

    Args:
        data (RDD[id, line]): The RDD to split.
        no_addr: Do not include the address.

    Returns:
        RDD[id, (addr, byte)] if `no_addr` is false (default).
        RDD[id, byte] if `no_addr` is true.
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    def split(line):
        try:
            (addr, *bytes) = line.split()
            bytes = [int(b, 16) for b in bytes]
            if no_addr: return bytes
            addr = int(addr, 16)
            return [(addr+i, b) for i,b in enumerate(bytes)]
        except ValueError:
            # ValueError occurs on invalid hex,
            # e.g. the missing byte symbol '??'.
            # For now, we discard the whole line. See #6.
            # https://github.com/dsp-uga/elizabeth/issues/6
            return []
    data = data.flatMapValues(split)

    data = data.persist()
    return data


def split_asm(data):
    '''Splits RDDs of the form `RDD[id, line]` into `RDD[id, (s, a, b, o, r)]`
    where `s` is the segment type, `a` is the address of the instruction, `b`
    is the big-end int value of the instruction, `o` is the opcode, and `r` is
    the rest of the instruction.

    The input is expected to be loaded from asm files.

    Args:
        data (RDD[id, line]): The RDD to split.

    Returns:
        RDD[id, (segment, addr, bytes, opcode, rest)]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext
    pattern = re.compile(r'\.([a-z]+):([0-9A-F]+)((?:\s[0-9A-F]{2})+)\s+([a-z]+)(?:\s+([^;]*))?')

    def match(x):
        (id, line) = x
        return pattern.match(line) is not None
    data = data.filter(match)

    def split(line):
        m = pattern.match(line)
        segment = m[1]
        addr = int(m[2], 16)
        bytes = parse_bytes(m[3])
        opcode = m[4]
        rest = m[5].strip()
        return (segment, addr, bytes, opcode, rest)
    data = data.mapValues(split)

    data = data.persist()
    return data



def parse_bytes(bytes):
    '''Parse strings like '0B AF 32' into big-end integers.
    '''
    val = 0
    for b in bytes.split():
        val = val << 8
        val = val + int(b, 16)
    return val
