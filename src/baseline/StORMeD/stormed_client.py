import json
import sys
from urllib import request


def query_stormed(data_to_query_, endpoint='parse', api_key=""):
    data = {
        'text': data_to_query_,
        'tagged': True,
        'key': api_key
    }
    data = json.dumps(data).encode("utf8")

    url = "https://stormed.inf.usi.ch/service/%s" % endpoint
    headers = {
        'Content-type': 'application/json',
        'Accept': 'text/plain',
        'Charset': 'UTF-8'
    }

    response = request.urlopen(request.Request(url, data=data, headers=headers))
    wrapper = json.loads(response.fp.read())
    if wrapper['status'] == "OK":
        return wrapper['status'], int(wrapper['quotaRemaining']), wrapper['result']
    else:
        return wrapper['status'], wrapper['message'], None


if __name__ == '__main__':
    api_key = sys.argv[1]
    data_to_query = ' '.join(sys.argv[2:])

    status, q_or_msg, result = query_stormed(data_to_query, api_key=api_key)
    if result is not None:
        for node in result:
            print(node['type'])
