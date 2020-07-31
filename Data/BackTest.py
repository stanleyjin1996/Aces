class BackTest:
    def __init__(self, data):
        self.data = data #delta
        self.regime = None
    
    def choose_best_seed(self, num):
        max_score = -float('inf')
        for i in range(50):
            model = GaussianHMM(n_components=2, n_iter = 1000, random_state = i)
            model.fit(num)
            temp = model.score(num)
            if temp > max_score:
                best_seed = i
                max_score = temp
        return best_seed
    
    def detect_regime(self, days_to_train=1000, rebalance=100, min_length=60):
        '''
        detect regime using data
        :param days_to_train: days of data used to train model for each time
        :param rebalance: frequency of re-estimate the model
        '''
        data = self.data
        seed = self.choose_best_seed(data.iloc[:days_to_train])
        new_model = GaussianHMM(n_components=2, n_iter = 1000, random_state = seed)
        new_model.fit(data.iloc[:days_to_train])
        regime = []
        
        flag = False
        for i in range(days_to_train, len(data)):
            if (i - rebalance + 1) % rebalance == 0:
                seed = self.choose_best_seed(data.iloc[:i])
                new_model = GaussianHMM(n_components=2, n_iter = 1000, random_state = seed)
                new_model.fit(data.iloc[:i])
                #check if regime definations are consistent
                pred_new = new_model.predict(data.iloc[:i])[days_to_train:]
                if (pred_new == np.array(regime)).sum() < 0.5 * (i - days_to_train):
                    flag = True
                else:
                    flag = False
            
            #predict states for a path of length days_to_train
            state = new_model.predict(data.iloc[:i+1])[-1]
            
            if flag:
                state = abs(1 - state)

            regime.append(state)
        regime = self.regime_filter(regime,min_length)
        regime = pd.DataFrame(regime, index = data.index[days_to_train:],columns = ['state'])
        self.regime = regime
        return regime
    
    def regime_filter(self,regime,min_length):
        
        for i in range(0,len(regime)-min_length,min_length):
            target = regime[i]
            if np.sum(regime[i:i+min_length]) != target * min_length:
                regime[i:i+min_length] = [target] * min_length
        return regime
    
    
    def regime_switch(self, regime):
        '''
        returns list of starting points of each regime
        :param regime: daily regime
        '''
        n = len(regime)
        init_points = [0]
        init_states = [regime.iloc[0].values[0]]
        for i in range(1,n):
            if (regime.iloc[i].values[0] != regime.iloc[i - 1]).values[0]:
                init_points.append(i)
                init_states.append(regime.iloc[i].values[0])
        
        init_points.append(n)
        #return init_points, init_states
        return init_points
    
    def plot_regime_color(self, df, regime, start):
        '''Plot of data versus regime'''
        regimelist = self.regime_switch(regime)
        y_max = df.max().max()
        curr_reg = start

        fig, ax = plt.subplots()
        for i in range(len(regimelist)-1):
            if curr_reg == 0:
                ax.axhspan(0, y_max*1.2, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                            facecolor='green', alpha=0.3) 
            else:
                ax.axhspan(0, y_max*1.2, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                            facecolor='red', alpha=0.5)
            curr_reg = abs(curr_reg - 1)

        fig.set_size_inches(20,9)   
        for col in df.columns:
            plt.plot(df[col],label=col)
        plt.legend()
        plt.ylabel('value')
        plt.xlabel('Year')
        plt.xlim([df.index[0], df.index[-1]])
        plt.ylim([0, y_max*1.2])
        plt.show()
