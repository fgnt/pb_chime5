class Dispatcher(dict):
    """Is basically a dict with a better error message on key error."""

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise KeyError(
                f'Invalid option {item!r}. Possible keys are {self.keys()!r}.'
            )
