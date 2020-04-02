import re

URI = re.compile(
    r'^(?:http|ftp|file|smtp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

EMAIL = re.compile(r"^([\w\.\-]+)@([\w\-]+)((\.(\w){2,3})+)$", re.IGNORECASE)


"""
This should consume lines such as:
diff --git a/<path> b/<path>
index 71bc177..fa9cdfe 100644
--- a/<path>
+++ b/<path>
"""
DIFF_HEADER = re.compile(
    r"(?:^diff --git a/.* b/.*)|"
    r"(?:^index [0-9a-f]{7}\.\.[0-9a-f]{7} [0-9]+)|"
    r"(?:^[+-]{3} [ab]/.*)", re.IGNORECASE
)


def is_URI(_input):
    return URI.match(_input) is not None


def is_email(_input):
    return EMAIL.match(_input) is not None


def is_diff_header(_input):
    return DIFF_HEADER.match(_input) is not None
