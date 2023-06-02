''' corpus cleaning... '''

quanjiao2b = ['……','。。。','。', '，', '；', '：', '？', '！', '“', '”', "‘", "’", "（", "）", '【', '】', '、']
banjiao = ['...','...','.', ',', ';', ':', '?', '!', '"', '"', "'", "'", "(", ")", '[', ']', ',']


def repl(data, fromjiao, tojiao):
    assert len(fromjiao)==len(tojiao)
    for i, j in zip(fromjiao, tojiao):
        # if i == '"':
        #     print()
        data = data.replace(i, j)
    return data

def process(line):
    '''char Clean, from data preprocessing scripts'''
    new_line = line.replace('\n', ' ') # remove \n
    puncs = [',', '.', ';', ':', '"', "'", "?", '!']
    p1 = [',', '.', ';', ':', "?", '!']
    p2 = ["'"]
    p3 = []
    # policy 1: xxx, xxx
    # policy 2: xxx'xxx
    # policy 3: 'xxx
    # p1 deal with: wrong space around pronounciation
    # first: clean all space, then add
    for _ in range(5):
        for p in p1:
            new_line = new_line.replace(' '+p+' ', p)
            new_line = new_line.replace(p+' ', p)
            new_line = new_line.replace(' '+p, p)

    for p in p1:
        new_line = new_line.replace(p, p+' ')
    # for ...:
    new_line = new_line.replace('. . . ', '... ')
    # p2 deal with: 's, 't
    wrong_samples = []
    for i in range(1, len(new_line)-2):
        if new_line[i]=="'" and new_line[i+1].isalpha() and new_line[i-1] ==' ' and new_line[i+2] ==' ':
            j = i-2
            while j>=1 and new_line[j]==' ':
                j-=1

            wrong_samples.append(new_line[j:i+3])
    wrong_samples.sort(key=lambda x:len(x), reverse=True)
    for w in wrong_samples:
        new_line = new_line.replace(w, w[0]+w[-3:])
    new_line = new_line.replace(" n't", "n't")
    # remove extra spaces at the end
    for k in range(len(new_line)-1, -1, -1):
        if new_line[k] != ' ':
            new_line = new_line[:k+1]
            break
    return new_line

def en_cleaning(data):
    d = repl(data, quanjiao2b, banjiao)
    d = process(d)
    new_d = d.replace('  ', ' ')
    while new_d != d:
        d = new_d
        new_d = d.replace('  ', ' ')
    return new_d


def clean_group(real, fake, real_qa=None, fake_qa=None, func=None):
    assert func is not None
    # print('Called')
    new_real, new_fake = list(), list()
    for i in range(len(real)):
        new_real.append(func(real[i]))
    for i in range(len(fake)):
        new_fake.append(func(fake[i]))
    if real_qa is None and fake_qa is None:
        return new_real, new_fake
    else:
        return new_real, new_fake, real_qa, fake_qa
    
def do_nothing(real, fake, real_qa=None, fake_qa=None):
    if real_qa is None and fake_qa is None:
        return real, fake
    else:
        return real, fake, real_qa, fake_qa
