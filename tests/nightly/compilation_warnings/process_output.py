import re
import sys
import operator

def process_output(command_output):
    warnings = {}
    regex = r"(.*):\swarning:\s(.*)"
    lines = command_output.split("\n")
    for line in lines[:-2]:
        matches = re.finditer(regex, line)
        for matchNum, match in enumerate(matches):
            try:
                warnings[match.group()] +=1
            except KeyError:
                warnings[match.group()] =1
    time = lines[-2]
    return time, warnings

def generate_stats(warnings):
    total_count = sum(warnings.values())
    sorted_warnings = sorted(warnings.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_warnings, total_count

def print_summary(time, warnings):
    sorted_warnings, total_count = generate_stats(warnings)
    print "START - Compilation warnings count"
    print total_count, 'warnings'
    print "END - Compilation warnings count"
    print 'START - Compilation warnings summary'
    print 'Time taken to compile:', time, 's'
    print 'Total number of warnings:', total_count, '\n'
    print 'Below is the list of unique warnings and the number of occurrences of that warning'
    for warning, count in sorted_warnings:
        print count, ': ', warning
    print 'END - Compilation warnings summary'

c_output = open(sys.argv[1],'r')
time, warnings = process_output(c_output.read())
print_summary(time, warnings)
