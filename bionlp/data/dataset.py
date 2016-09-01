class Dataset:
    def __init__(self,DocumentList,desc="None"):
        self.description=desc
        self.value=DocumentList
        self.active=[]          #set of active attributes used for Machine Learning
        self.passive=[]       #set of passive attributes not used for trainig. e.g. gold-label, char offsets.
        self.delayed=[]       #set of active attributes which require too much memory. So they are only computed in the tagger section.
        self.attr={}

    def __str__(self):
        s="Dataset description = {0}\n active Attributes {1}, passive Attributes {2}, delayed attributes {3} , Number of Documents {4}".format(self.description,self.active,self.passive,self.delayed,self.value.__len__())
        return s
