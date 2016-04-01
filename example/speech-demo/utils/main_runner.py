import sys
import json
import pprint
import logging
import os
import validictory

import utils

def run_main(main_func, schema, args):
	if len(args) < 2:
		print "Usage: " + args[0] + " <json config file>"
		print "For possible configs: " + args[0] + " help"
		sys.exit(1)
	if args[1].lower() in ["--h", "-h", "help", "-help", "--help"]:
		#pprint.pprint(schema)
		print json.dumps(schema, indent=2)
		sys.exit(0)

	arguments = {}
	for i in xrange(1, len(args)):
		try:
			config = args[i]
			print >> sys.stderr, "Merging %s into configuration" %(config,)
			arguments.update(json.load(open(config)))
		except Exception, e:
			print e
			sys.exit(1)

	validictory.validate(arguments, schema)
	logging_ini = None
	if "logging_ini" in arguments:
		logging_ini = arguments["logging_ini"]
	utils.setup_logger(logging_ini)
	main_func(arguments)