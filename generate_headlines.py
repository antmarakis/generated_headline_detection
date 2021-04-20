from transformers import pipeline
generator = pipeline('text-generation', model='PATH_TO_MODEL')

def parse(txt):
    # chr(999) and chr(1000) are special start and end tokens
    return txt.split(chr(999))[1].split(chr(1000))[0].strip()

for i in range(N): # generate in batches of N
    print('gen {}'.format(i))
    seqs = [seq['generated_text'] for seq in generator(chr(999), max_length=30, num_return_sequences=10, lowercase=False)]

    seqs = [parse(seq) for seq in seqs if chr(1000) in seq] # only keep headlines where the end token has been generated
    print(seqs)

    f = open('OUTFILE.txt', 'a+')
    f.write('\n'.join(seqs))
    f.close()
