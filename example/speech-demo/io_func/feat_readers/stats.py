import numpy

class _StreamVariance(object):

    def __init__(self,nCols):
        self.n    = 0;
        self.mean = numpy.zeros(nCols)
        self.M2   = numpy.zeros(nCols)

    def AddX(self,value):
        # do not operate in the same way when the input is an 1
        # dimension array or a 2 dimension array.  Maybe there is
        # a better way to handle that
        if len(value.shape) == 2:
            for x in value:
                self.n     = self.n+1
                delta      = x-self.mean
                self.mean  = self.mean+delta/self.n
                self.M2    = self.M2+delta*(x-self.mean)
        elif len(value.shape) == 1:
            self.n     = self.n+1
            delta      = value-self.mean
            self.mean  = self.mean+delta/self.n
            self.M2    = self.M2+delta*(value-self.mean)
        else:
            msg = 'Only 1D and 2D array are supported'
            raise Exception(msg)

    def GetMean(self):
        return self.mean

    def GetVariance(self):
        return self.M2/(self.n-1)

    def GetInvStandardDeviation(self):
        return 1.0/(numpy.sqrt(self.M2/(self.n-1)))

    def GetNumberOfSamples(self):
        return self.n

class FeatureStats(object):

    def __init__(self):
        self.mean           = numpy.zeros(1,)
        self.invStd         = numpy.zeros(1,)
        self.populationSize = 0
        self.dim            = None

    def GetMean(self):
        return self.mean

    def GetVariance(self):
        return numpy.power(self.GetStd(),2)

    def GetStd(self):
        return 1.0/self.invStd

    def GetInvStd(self):
        return self.invStd

    """

    def GetStatsFromList(self,fileList,featureFileHandler):
        stats = None

        for featureFile,label in featureList.FeatureList(fileList):
            if stats == None:
                self.dim = self.getDimFromFile(featureFile,featureFileHandler)
                stats    = _StreamVariance(self.dim)

            samples = featureFileHandler.Read(featureFile)

            print 'Process file : "{}"'.format(featureFile)
            stats.AddX(samples)

        print 'Read {} samples'.format(stats.GetNumberOfSamples())
        self.mean           = stats.GetMean()
        self.invStd         = stats.GetInvStandardDeviation()
        self.populationSize = stats.GetNumberOfSamples()

        return (self.mean,self.invStd)

    def GetStatsFromFile(self,featureFile,featureFileHandler):
        self.dim = self.getDimFromFile(featureFile,featureFileHandler)
        stats = _StreamVariance(self.dim)

        samples = featureFileHandler.Read(featureFile)
        stats.AddX(samples)
        self.mean           = stats.GetMean()
        self.invStd         = stats.GetInvStandardDeviation()
        self.populationSize = stats.GetNumberOfSamples()

        return (self.mean,self.invStd)

    def getDimFromFile(self,featureFile,featureFileHandler):
        return featureFileHandler.GetDim(featureFile)

    """

    def Load(self,filename):
        with open(filename,"rb") as f:
            dt = numpy.dtype([('magicNumber',(numpy.int32,1)),('numSamples',(numpy.int32,1)),('dim',(numpy.int32,1))])
            header = numpy.fromfile(f,dt,count=1)

            if header[0]['magicNumber'] != 21812:
                msg = 'File {} is not a stat file (wrong magic number)'
                raise Exception(msg)

            self.populationsize = header[0]['numSamples']
            dim = header[0]['dim']

            dt = numpy.dtype([('stats',(numpy.float32,dim))])
            self.mean    = numpy.fromfile(f,dt,count=1)[0]['stats']
            self.invStd  = numpy.fromfile(f,dt,count=1)[0]['stats']

    def Save(self,filename):
        with open(filename,'wb') as f:
            dt = numpy.dtype([('magicNumber',(numpy.int32,1)),('numSamples',(numpy.int32,1)),('dim',(numpy.int32,1))])
            header=numpy.zeros((1,),dtype=dt)
            header[0]['magicNumber']=21812
            header[0]['numSamples']=self.populationSize
            header[0]['dim']=self.mean.shape[0]
            header.tofile(f)

            self.mean.astype(numpy.float32).tofile(f)
            self.invStd.astype(numpy.float32).tofile(f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Print the mean and standard deviation from a stat file',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',help="Name of the stat file")
    args = parser.parse_args()
    featureStats = FeatureStats()
    featureStats.Load(args.filename)

    numpy.set_printoptions(threshold='nan')
    print("THIS IS THE MEAN: ")
    print(featureStats.GetMean())
    print("THIS IS THE INVERSE STD: ")
    print(featureStats.GetInvStd())


