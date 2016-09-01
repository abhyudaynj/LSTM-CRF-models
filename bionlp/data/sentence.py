class Sentence:
    def __init__(self,TokenList,id=-1):
        self.id=id
        self.attr={}
        self.attr['document']=TokenList[0].attr['document']
        self.value=TokenList

    def get_text(self):
        return ' '.join(token.value for token in self.value)

    def get_list(self):
        return [token.value for token in self.value]


    def __str__(self):
        s ="Sentence id: {0}, value: {1}, attributes: {2}".format(self.id,self.get_text(),self.attr)
        return s
