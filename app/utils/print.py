import pprint


class TruncatingPrettyPrinter(pprint.PrettyPrinter):
    def __init__(self, *args, truncate_at=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncate_at = truncate_at

    def _format(self, obj, stream, indent, allowance, context, level):
        if isinstance(obj, list) and len(obj) > self.truncate_at:
            obj = self._truncate_list(obj)
        super()._format(obj, stream, indent, allowance, context, level)

    def _truncate_list(self, lst):
        start = self.truncate_at // 2
        end = self.truncate_at - start
        return lst[:start] + ["..."] + lst[-end:]


def truncating_pformat(obj, truncate_at=10, *args, **kwargs):
    printer = TruncatingPrettyPrinter(truncate_at=truncate_at, *args, **kwargs)
    return printer.pformat(obj)
