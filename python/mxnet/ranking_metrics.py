"""
A function that implements ranking metrics 
(precision, recall, coverage and converted coverage) 
for a given position K
"""

def metrics_at_k(Y,Yhat, K):
	"""
	Parameters
	----------
	Y 	: dictionary with key = sample index and value = list of positive indices of features
 	Yhat: dict wit key = sample index and value = ORDERED list of indices of features, according to some score
	K 	: position at which to compute score
	
	Returns
	-------
	pre 		: precision at K
	rec 		: recall at K
	convcoverage: converted coverage at K
	coverage 	: coverage at K

	Examples
    --------
    >>>Ytrue = {1: [1,2,3,4]}
	>>>Yhat = {1:[1,2,3,4,5,6,7,8,9]}
	>>>k = 2
	>>>print(metrics_at_k(Ytrue,Yhat,k))

	"""
	pre = 0.0
	rec = 0.0
	allhits = set()
	allpreds = set()
	for k in Yhat.keys():
		truepos = set(Y[k])
		predpos = Yhat[k]
		predpos = set(predpos[:K]) # only retain top K
		allpreds = allpreds.union(predpos)
		hits = predpos.intersection(truepos)
		pre += (len(hits)*1.0)/K
		allhits = allhits.union(hits)
		rec += (len(hits)*1.0)/len(truepos)
	convcoverage = len(allhits)
	coverage = len(allpreds)
	return pre/len(Yhat),rec/len(Yhat),convcoverage,coverage