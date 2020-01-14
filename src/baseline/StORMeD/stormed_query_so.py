import fileinput
import json
import sys

import xmltodict
from bs4 import BeautifulSoup

from .stormed_client import query_stormed

HTML_PARSER = "html5lib"


def parse_stackoverflow_posts(file_location_, for_stormed=False):
    with fileinput.input(file_location_, openhook=fileinput.hook_encoded("utf-8")) as f_:
        for line in f_:
            line = line.strip()
            if line.startswith("<row"):
                row = xmltodict.parse(line)['row']
                if for_stormed and ('@Tags' not in row.keys() or '<java>' not in row['@Tags']):
                    continue
                yield row['@Body']


if __name__ == '__main__':
    file_location = sys.argv[1]
    starting_offset = int(sys.argv[2])
    api_key = sys.argv[3]
    for pos, post in enumerate(parse_stackoverflow_posts(file_location)):
        if pos < starting_offset:
            continue
        status, quota_or_msg, maybe_result = query_stormed(BeautifulSoup(post, HTML_PARSER).text, api_key=api_ley)
        if status == 'OK':
            quota = quota_or_msg
            with open('./data/stormed/stormed_%d.json' % pos, 'w') as f:
                f.write(json.dumps(maybe_result))
            if quota == 0:
                print('Stopping at position %d, daily quota reached!' % pos)
                exit(0)
        else:
            print('Encountered error response from StORMeD!')
