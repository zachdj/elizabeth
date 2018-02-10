import re

import elizabeth


def hash_to_url(x=None, base='gs', kind='bytes'):
    '''Returns a function mapping document hashes to Google Storage URLs.

    The API for this function is fancy. It can be used directly as a closure:

        >>> my_rdd.map(hash_to_url)

    Or it can be used to return a closure with different defaults:

        >>> my_rdd.map(hash_to_url(base='https', kind='asm'))

    The first form is convenient for interactive use, but may be confusing or
    unclear for production use. Prefer the second form in that case.

    Args:
        x (str):
            The hash identifying a document instance.
        base (str):
            The base of the URL or local path to the data path. If it is the
            special string 'https', then the https base URL is used. Likewise
            if it is the special string 'gs', then the Google Storage base URL
            is used.
        kind (str):
            The kind of file to use, either 'bytes' or 'asm'.

    Returns:
        If `x` is given, returns the URL to the document.
        If `x` is None, returns a closure that takes a hash and returns a URL.
    '''
    if base == 'https': base = 'https://storage.googleapis.com/uga-dsp/project2/data'
    if base == 'gs': base = 'gs://uga-dsp/project2/data'

    closure = lambda x: f'{base}/{kind}/{x}.{kind}'

    if x is None:
        return closure
    else:
        return closure(x)


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
            The base of the URL or local path to the data path. If it is the
            special string 'https', then the https base URL is used. Likewise
            if it is the special string 'gs', then the Google Storage base URL
            is used.
        kind (str):
            The kind of file to use, either 'bytes' or 'asm'.

    Returns:
        RDD[id, line]
    '''
    spark = elizabeth.session()
    ctx = spark.sparkContext

    # Cast to str to support pathlib.Path etc.
    manifest = str(manifest)

    # Read the manifest as an RDD[id, url].
    manifest = ctx.textFile(manifest)                                 # RDD[hash]
    manifest = manifest.zipWithIndex()                                # RDD[hash, id]
    manifest = manifest.map(lambda x: (x[1], x[0]))                   # RDD[id, hash]
    manifest = manifest.mapValues(hash_to_url(base=base, kind=kind))  # RDD[id, url]

    # Load each URL as a separate RDD[line], then combine to RDD[id, line].
    id_mapper = lambda id: lambda x: (id, x)
    data = {id: ctx.textFile(url) for id, url in manifest.toLocalIterator()}  # {id: RDD[line]}
    data = [rdd.map(id_mapper(id)) for id, rdd in data.items()]               # [RDD[id, line]]
    data = ctx.union(data)                                                    # RDD[id, line]

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
