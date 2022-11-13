from urllib.request import urlopen

HOST = 'http://localhost'
PORT = '80'

def get_data():
    return urlopen(HOST+ ':' + PORT + '/').status

if __name__=='__main__':
    print(get_data())