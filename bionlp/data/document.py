class Document:
    def __init__(self,SentenceList,id=-1):
        self.id=id
        self.value=SentenceList
        self.attr={}

    def __str__(self):
        s ="Document id: {0}, value: {1}, attributes: {2}".format(self.id,self.value,self.attr)
        return s
