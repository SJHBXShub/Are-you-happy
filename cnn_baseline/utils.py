# this file is used to define kinds of utility functions

def aveResult(result_array_list):
    final = sum(result_array_list)/len(result_array_list)
    final = pd.DataFrame(final)
    final.to_csv('result_.txt', index = None, header = None)