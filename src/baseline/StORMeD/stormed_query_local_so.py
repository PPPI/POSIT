import json
import os
import sys
import urllib.error

# noinspection PyProtectedMember
from bs4 import BeautifulSoup, Tag

from .stormed_client import query_stormed

HTML_PARSER = "html5lib"


def process_single_rev(target_data_, post_id, rev_, api_key):
    with open(os.path.join(os.path.dirname(target_data_), 'SO_Posts', post_id, '%d.html' % rev_)) as f_:
        html_ = f_.read()
    row = ''
    for elem in BeautifulSoup(html_, HTML_PARSER).find('div').contents:
        if isinstance(elem, Tag):
            if elem.name == 'pre':
                row += str(elem.next_element)
            elif elem.name == 'code':
                row += str(elem)
            else:
                row += elem.text
        else:
            row += str(elem)
    row = row.strip()
    no_tag_row = BeautifulSoup(html_, HTML_PARSER).find('div').text
    status, quota_or_msg, maybe_result = query_stormed(no_tag_row, api_key=api_key)
    os.makedirs('./results/stormed/paired_posts/%s/' % post_id, exist_ok=True)
    if status == 'OK':
        quota = quota_or_msg
        with open('./results/stormed/paired_posts/%s/%d.json' % (post_id, rev_), 'w') as f_:
            f_.write(json.dumps(maybe_result))
        if quota == 0:
            print('Stopping at position rev: %s, post: %s, daily quota reached!' % (post_id, rev_))
            exit(0)
    else:
        print('Encountered error response from StORMeD!')
    try:
        status, quota_or_msg, maybe_result = query_stormed(row, endpoint='tagger', api_key=api_key)
        if status == 'OK':
            quota = quota_or_msg
            with open('./results/stormed/paired_posts/%s/%d.html' % (post_id, rev_), 'w') as f_:
                f_.write(json.dumps(maybe_result))
            if quota == 0:
                print('Stopping at position rev: %s, post: %s, daily quota reached!' % (post_id, rev_))
                exit(0)
        else:
            print('Encountered error response from StORMeD!')
    except urllib.error.HTTPError:
        print('Server error for Tagger!')


if __name__ == '__main__':
    target_data = sys.argv[1]
    api_key = sys.argv[2]
    with open(target_data) as f:
        lines_and_revs = [l.strip().split(',') for l in f.readlines()][1:]
    for postId, rev in lines_and_revs:
        process_single_rev(target_data, postId, int(rev) - 1, api_key)
        process_single_rev(target_data, postId, int(rev), api_key)
