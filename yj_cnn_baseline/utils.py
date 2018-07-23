#s this file is used to define kinds of utility functions

class MyFunction(object):
	# Read Embedding File
    def read_embedding(filename):
        embed = {}
        for line in open(filename):
            line = line.strip().split()
            embed[str(line[0])] = list(map(float, line[1:]))
        print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
        return embed

def aveResult(result_array_list):
    final = sum(result_array_list)/len(result_array_list)
    final = pd.DataFrame(final)
    final.to_csv('result_.txt', index = None, header = None)
