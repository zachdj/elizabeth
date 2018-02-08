def hash_to_url(x=None, http=False, asm=False):
    '''Returns a function mapping document hashes to Google Storage URLs.

    The API for this function is fancy. It can be used directly as a closure:

        >>> my_rdd.map(hash_to_url)

    Or it can be used to return a closure with different defaults:

        >>> my_rdd.map(hash_to_url(http=True, asm=True))

    The first form is convenient for interactive use, but may be confusing or
    unclear for production use. Prefer the second form in that case.

    Args:
        x (str): The hash identifying a document instance.
        http (bool): Use https instead of gs.
        asm (bool): Download the assembly instead of bytes files.

    Returns:
        If `x` is given, returns the URL to the document.
        If `x` is None, returns a closure that takes a hash and returns a URL.
    '''
    if http and asm:
        closure = lambda x: f'https://storage.googleapis.com/uga-dsp/project2/data/asm/{x}.asm'
    elif http:
        closure = lambda x: f'https://storage.googleapis.com/uga-dsp/project2/data/bytes/{x}.bytes'
    elif asm:
        closure = lambda x: f'gs://uga-dsp/project2/data/asm/{x}.asm'
    else:
        closure = lambda x: f'gs://uga-dsp/project2/data/bytes/{x}.bytes'

    if x is None:
        return closure
    else:
        return closure(x)


def load_data(ctx, manifest, http=False, asm=False):
    '''Load data from a manifest file into an RDD.

    A manifest file gives the hash identifying each document on separate lines.

    The returned RDD is of the form `RDD[id, line]` where `id` uniquely
    identifies the document and `line` is a line of the document as a string.

    Note that the document ID is _not_ the same as the hash. The ID is
    guaranteed to uniquely identify one document and the order of the IDs is
    guaranteed to match the order given in the manifest file.

    Args:
        ctx: The SparkContext in which to operate.
        manifest: Path or URL of the manifest file.
        http: Download malware files over http instead of gs.
        asm: Download assembly instead of bytes files.

    Returns:
        RDD[id, line]
    '''
    # Cast to str to support pathlib.Path etc.
    manifest = str(manifest)

    # Read the manifest as an RDD[id, url].
    manifest = ctx.textFile(manifest)                               # RDD[hash]
    manifest = manifest.zipWithIndex()                              # RDD[hash, id]
    manifest = manifest.map(lambda x: (x[1], x[0]))                 # RDD[id, hash]
    manifest = manifest.mapValues(hash_to_url(http=http, asm=asm))  # RDD[id, url]

    # Load each URL as a separate RDD[line], then combine to RDD[id, line].
    data = {id: ctx.textFile(url) for id, url in manifest.toLocalIterator()}  # {id: RDD[line]}
    data = [rdd.map(lambda x: (id, x)) for id, rdd in data.items()]           # [RDD[id, line]]
    data = ctx.union(data)                                                    # RDD[id, line]

    return data


def load_labels(ctx, labels):
    '''Load labels from a label file into an RDD.

    A label file corresponds to a manifest file and each line gives the label
    of the document on the same line of the manifest file.

    The returned RDD is of the form `RDD[id, label]` where `id` uniquely
    identifies the document and `label` is a label for the document.

    Note that the document ID is _not_ the same as the hash. The ID of a
    document is guaranteed to match the ID returned by `load_data` from the
    corresponding manifest file.

    Args:
        ctx: The SparkContext in which to operate.
        labels: Path or URL of the label file.

    Returns:
        RDD[id, label]
    '''
    # Cast to str to support pathlib.Path etc.
    labels = str(labels)

    # Read the labels as an RDD[id, url].
    labels = ctx.textFile(labels)                # RDD[label]
    labels = ctx.zipWithIndex(labels)            # RDD[label, id]
    labels = labels.map(lambda x: (x[1], x[0]))  # RDD[id, label]

    return labels


def split_bytes(ctx, data, no_addr=False):
    '''Splits RDDs of the form `RDD[id, line]` into `RDD[id, (addr, byte)]`
    where `addr` is the address of the byte, and `byte` is the value.

    Args:
        data (RDD[id, line]): The RDD to split.
        no_addr: Do not include the address.

    Returns:
        RDD[id, (addr, byte)] if `no_addr` is false (default).
        RDD[id, byte] if `no_addr` is true.
    '''
    def split(line):
        (addr, *bytes) = data.split()
        bytes = [int(b, 16) for b in bytes]
        if no_addr: return bytes
        addr = int(addr, 16)
        bytes = [(addr+i, b) for i,b in enumerate(bytes)]
    return data.flatMapValues(split)
