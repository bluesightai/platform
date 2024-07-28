import random
import string


def random_string(k: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=k))
