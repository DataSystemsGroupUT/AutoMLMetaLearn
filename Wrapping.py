class ffun():
    def __init__(self, X_train,y_train,randomized_search):


        self.X_train= X_train
        self.y_train=y_train
        self.randomized_search=randomized_search
  

    def fun(self):
        return self.randomized_search.fit(self.X_train,self.y_train)
