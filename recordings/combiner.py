from pathlib import Path

p = Path('.')
text_files = p.glob('*.txt')
output = p / 'output.txt'

texts = []

for f in text_files:
    with f.open() as ff:
        l = [ x.strip() for x in ff.readlines() ]
        texts.append(list(filter(None, l)))

print(len(texts))
print(texts[0])

with output.open('w') as o:
    print(o)
    for q in texts:
        if len(q) == 1:
            o.write( "Q: {} \nR: -\n\n".format( q[0] ) )
            print("len is {}".format(len(q)))
        if len(q) > 1:
            o.write( "Q: {}\nR: {}\n\n".format( q[0], q[1] ))
        #print('Q: {} /n R: {}'.format(q[0], q[1]))
