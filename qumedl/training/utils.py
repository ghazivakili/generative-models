import requests


def generate_random_name(
    n: int = 2,
    prefix: str | None = None,
    suffix: str | None = None,
    url_tempalte: str = "https://random-word-api.vercel.app/api?words={n}",
) -> str:
    """Generates a random name with the given number of words.

    Args:
        n (int, optional): Number of words. Defaults to 2.
        prefix (str | None, optional): Prefix. Defaults to None.
        suffix (str | None, optional): Suffix. Defaults to None.
        url_tempalte (str, optional): URL template. Defaults to "https://random-word-api.vercel.app/api?words={n}".
            Template should contain a single {n} placeholder for the number of words.

    Returns:
        str: Random name in the format: prefix-{{word0-word1-...-}}-suffix
    """
    url = url_tempalte.format(n=n)

    words = list(requests.get(url).json())
    random_string = "-".join(words)

    if prefix is not None:
        random_string = f"{prefix}-{random_string}"

    if suffix is not None:
        random_string = f"{random_string}-{suffix}"

    return random_string
