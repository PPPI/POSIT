import json
import os
import sys
import time
import urllib.error

from nltk import pos_tag, flatten

from .stormed_client import query_stormed
from ..classification import parse_file, unpack_data

HTML_PARSER = "html5lib"


def chunk(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


if __name__ == '__main__':
    dataset = sys.argv[1]
    api_key = sys.argv[2]
    endpoint = 'parse'
    file_location_t = './data/corpora/%s/corpus/train.txt' % dataset
    file_location_d = './data/corpora/%s/corpus/dev.txt' % dataset
    file_location_e = './data/corpora/%s/corpus/eval_stormed.txt' % dataset

    W, X, y, z = unpack_data(parse_file(file_location_t))

    W_test, X_test, y_test, z_test = unpack_data(parse_file(file_location_e))
    W_test = flatten(W_test)
    W_test = chunk(W_test, len(W_test) // 1000 + 1)
    X_test = flatten(X_test)
    X_test = chunk(X_test, len(X_test) // 1000 + 1)
    y_test = flatten(y_test)
    y_test = chunk(y_test, len(y_test) // 1000 + 1)
    z_test = flatten(z_test)
    z_test = chunk(z_test, len(z_test) // 1000 + 1)
    # Perform one tag here to pre-load the tags
    pos_tag(W[0], tagset="universal")

    os.makedirs('./data/java/stormed_eval_%s' % endpoint, exist_ok=True)
    for pos in range(len(W_test)):
        msg = W_test[pos]

        expected = z_test[pos] if endpoint == 'tagger' else y_test[pos]
        if endpoint == 'parse':
            msg = zip(W_test[pos], z_test[pos], y_test[pos])
            inCode = False
            query_str = ''
            for tok, lid, tag in msg:
                isCode = lid == 1

                if not isCode and inCode:
                    inCode = False
                    query_str += '</code>'
                if isCode and not inCode:
                    inCode = True
                    query_str += '<code>'

                if tag != '.':
                    query_str += ' '
                query_str += tok

            msg_query = query_str
        else:
            msg_query = ' '.join(msg)
        retry = True
        max_retry = 50
        retry_n = 0
        while retry:
            try:
                status, quota_or_msg, maybe_result = query_stormed(msg_query, endpoint=endpoint, api_key=api_key)
                retry = False
            except urllib.error.HTTPError as e:
                if (e.code > 500) and (retry_n <= max_retry):
                    retry_n += 1
                    time.sleep(10)
                else:
                    print('Encountered error response from StORMeD!')

        if status == 'OK':
            quota = quota_or_msg
            with open('./data/java/stormed_eval_%s/stormed_%d.json' % (endpoint, pos), 'w') as f:
                f.write(json.dumps(maybe_result))
            with open('./data/java/stormed_eval_%s/stormed_%d_expected' % (endpoint, pos)
                      + ('' if endpoint == 'tagger' else '_tags') + '.json', 'w') as f:
                f.write(json.dumps(expected))
            if quota == 0:
                print('Stopping at position %d, daily quota reached!' % pos)
                exit(0)
        else:
            print('Encountered error response from StORMeD!')
