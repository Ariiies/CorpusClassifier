from numpy import asarray as array, log10


## the fusion between corpy and tf-idf
class DocuMetrics:

    def __init__(self, corpus):
        self.__corpus = corpus
        self.__vocabulary = sorted(set(' '.join(self.__corpus).split()))
        self.__data = self.__generate_word_data()

    def get_vocabulary(self):  
        return self.__vocabulary
  
    def __generate_word_data(self):
        """Generate word data for the entire corpus"""
        data = {}
        for ele in self.__vocabulary:
            data[ele], total, ndoc = {}, 0, 0
            for i in range(1, len(self.__corpus) + 1):
                flag, count, text = True, 0, self.__corpus[i - 1].split()
                for word in text:
                    if ele == word:
                        count += 1
                        total += 1
                        flag = False
                        data[ele].update({f'doc{i}': count})
                if flag:
                    data[ele].update({f'doc{i}': 0})
            data[ele].update({'total': total})
            for l in range(1, len(self.__corpus) + 1): 
                if data[ele][f'doc{l}'] != 0:
                    ndoc += 1
            data[ele].update({'exist in': ndoc})
        return data

    def get_word_data(self):
        return self.__data

    
    @property
    def TF(self): 
        tf = []
        for i in range(len(self.__corpus)):
            tfe = []
            for word in self.__vocabulary:
                tfe.append(self.__data[word][f'doc{i+1}'] / len(self.__corpus[i].split()))
            tf.append(tfe)
        return array(tf)
    @property
    def IDF(self): 
        idf = []
        n_docs = len(self.__corpus)
        for word in self.__vocabulary:
            idf_value = log10(n_docs / self.__data[word]['exist in'])
            idf.append(idf_value)
        return array(idf)
    @property
    def TF_IDF(self):
        return self.TF * self.IDF
