import os


class DataDir(str):
    """
    Append string class with join, for easier calling.

    Example:

        >>> import os
        >>> os.environ['NT_TESTING_DIR'] = '/net'
        >>> from nt.io.data_dir import testing as data_dir
        >>> data_dir.join('a', 'b')
        '/net/a/b'
        >>> data_dir
        '/net'
        >>> os.path.join(data_dir, '2')
        '/net/2'
        >>> data_dir('2')
        '/net/2'
        >>> type(data_dir.join('a', 'b'))
        <class 'str'>
    """

    def join(self, *args):
        return os.path.join(self, *args)

    def __call__(self, *args):
        return os.path.join(self, *args)


testing = DataDir(os.getenv(
    'NT_TESTING_DIR',
    '/net/storage/python_unittest_data'
))
timit = DataDir(os.getenv(
    'NT_TIMIT_DIR',
    '/net/speechdb/timit'
))

database_jsons = DataDir(os.getenv(
    'NT_DATABASE_JSONS_DIR',
    '/net/storage/database_jsons'
))
chime = DataDir(os.getenv(
    'NT_CHIME_DIR',
    '/net/ssd/2015/chime'
))

kaldi_root = DataDir(os.getenv(
    'KALDI_ROOT',
    '/Users/jahn/kaldi'
))

if __name__ == "__main__":
    import doctest

    doctest.testmod()
