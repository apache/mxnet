import numpy
import os

class ByteOrder:
	LittleEndian, BigEndian    = range(2)

class FeatureException(Exception):
	def __init__(self,msg):
		self.msg = msg
	def __str__(self):
		return repr(self.msg)

def ReadLabel(filename):
	labels = numpy.loadtxt(filename, ndmin=1)
	return labels.astype(numpy.int32)

class BaseReader():
	def __init__(self, featureFile, labelFile, byteOrder=None):
		self.byteOrder = byteOrder
		self.featureFile = featureFile
		self.labelFile = labelFile
		self.done = False

	def _markDone(self):
		self.done = True

	def IsDone(self):
		return self.done

	def Read(self):
		pass

	def Cleanup(self):
		pass

	# no slashes or weird characters
	def GetUttId(self):
		return os.path.basename(self.featureFile)

def getReader(fileformat, featureFile, labelFile):
	if fileformat.lower() == 'htk':
		import reader_htk
		return reader_htk.htkReader(featureFile, labelFile, ByteOrder.BigEndian)
	elif fileformat.lower() == 'htk_little':
		import reader_htk
		return reader_htk.htkReader(featureFile, labelFile, ByteOrder.LittleEndian)
	elif fileformat.lower() == 'bvec':
		import reader_bvec
		return reader_bvec.bvecReader(featureFile, labelFile)
	elif fileformat.lower() == 'atrack':
		import reader_atrack
		return reader_atrack.atrackReader(featureFile, labelFile)
	elif fileformat.lower() == 'kaldi':
		import reader_kaldi
		return reader_kaldi.kaldiReader(featureFile, labelFile)
	else:
		msg = "Error: Specified format '{}' is not supported".format(fileformat)
		raise Exception(msg)