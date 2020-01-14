from bs4 import BeautifulSoup
import random

try:
    import urllib.request as urllib
except ImportError:
    import urllib


def parse_mail(mail_url):
    start_tag = '<!--X-Body-of-Message-->'
    end_tag = '<!--X-Body-of-Message-End-->'
    with urllib.urlopen(mail_url) as f:
        html = str(f.read())
    html = html.split(start_tag)[-1]
    html = html.split(end_tag)[0]
    return BeautifulSoup(html, 'html5lib').text


def find_mails(start_url):
    with urllib.urlopen(start_url) as f:
        soup = BeautifulSoup(f.read(), 'html5lib')
    return [start_url[:-10] + mail.strong.a['href'] for mail in soup.find_all('li') if mail.strong is not None and mail.strong.a is not None and 'href' in mail.strong.a.attrs]


if __name__ == '__main__':
    seed_url = 'http://lkml.iu.edu/hypermail/linux/kernel/1808.3/index.html'
    mails = find_mails(seed_url)
    random.seed(42)
    mails = random.sample(mails, 30)
    with open('linux_mail.txt', 'w') as f:
        for mail in mails:
            text = parse_mail(mail)
            f.write(text)
            f.write('\n\n' + ''.join(['_'] * 80) + '\n\n')
    