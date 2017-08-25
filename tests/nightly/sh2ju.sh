#!/bin/sh
### Copyright 2010 Manuel Carrasco Moñino. (manolo at apache.org)
###
### Licensed under the Apache License, Version 2.0.
### You may obtain a copy of it at
### http://www.apache.org/licenses/LICENSE-2.0

###
### A library for shell scripts which creates reports in jUnit format.
### These reports can be used in Jenkins, or any other CI.
###
### Usage:
###     - Include this file in your shell script
###     - Use juLog to call your command any time you want to produce a new report
###        Usage:   juLog <options> command arguments
###           options:
###             -name="TestName" : the test name which will be shown in the junit report
###             -error="RegExp"  : a regexp which sets the test as failure when the output matches it
###             -ierror="RegExp" : same as -error but case insensitive
###     - Junit reports are left in the folder 'result' under the directory where the script is executed.
###     - Configure Jenkins to parse junit files from the generated folder
###

asserts=00; errors=0; total=0; content=""
date=`which gdate || which date`

# create output folder
juDIR=`pwd`/results
mkdir -p "$juDIR" || exit

# The name of the suite is calculated based in your script name
suite=`basename $0 | sed -e 's/.sh$//' | tr "." "_"`

# A wrapper for the eval method witch allows catching seg-faults and use tee
errfile=/tmp/evErr.$$.log
eVal() {
  eval "$1"
  echo $? | tr -d "\n" >$errfile
}

# Method to clean old tests
juLogClean() {
  echo "+++ Removing old junit reports from: $juDIR "
  rm -f "$juDIR"/TEST-*
}

# Execute a command and record its results
juLog() {

  # parse arguments
  ya=""; icase=""
  while [ -z "$ya" ]; do
    case "$1" in
  	  -name=*)   name=$asserts-`echo "$1" | sed -e 's/-name=//'`;   shift;;
      -ierror=*) ereg=`echo "$1" | sed -e 's/-ierror=//'`; icase="-i"; shift;;
      -error=*)  ereg=`echo "$1" | sed -e 's/-error=//'`;  shift;;
      *)         ya=1;;
    esac
  done

  # use first arg as name if it was not given
  if [ -z "$name" ]; then
    name="$asserts-$1"
    shift
  fi

  # calculate command to eval
  [ -z "$1" ] && return
  cmd="$1"; shift
  while [ -n "$1" ]
  do
     cmd="$cmd \"$1\""
     shift
  done

  # eval the command sending output to a file
  outf=/var/tmp/ju$$.txt
  >$outf
  echo ""                         | tee -a $outf
  echo "+++ Running case: $name " | tee -a $outf
  echo "+++ working dir: "`pwd`           | tee -a $outf
  echo "+++ command: $cmd"            | tee -a $outf
  ini=`$date +%s.%N`
  eVal "$cmd" 2>&1                | tee -a $outf
  evErr=`cat $errfile`
  rm -f $errfile
  end=`date +%s.%N`
  echo "+++ exit code: $evErr"        | tee -a $outf

  # set the appropriate error, based in the exit code and the regex
  [ $evErr != 0 ] && err=1 || err=0
  out=`cat $outf | sed -e 's/^\([^+]\)/| \1/g'`
  if [ $err = 0 -a -n "$ereg" ]; then
      H=`echo "$out" | egrep $icase "$ereg"`
      [ -n "$H" ] && err=1
  fi
  echo "+++ error: $err"         | tee -a $outf
  rm -f $outf

  # calculate vars
  asserts=`expr $asserts + 1`
  asserts=`printf "%.2d" $asserts`
  errors=`expr $errors + $err`
  time=`echo "$end - $ini" | bc -l`
  total=`echo "$total + $time" | bc -l`

  # write the junit xml report
  ## failure tag
  [ $err = 0 ] && failure="" || failure="
      <failure type=\"ScriptError\" message=\"Script Error\"></failure>
  "
  ## testcase tag
  content="$content
    <testcase assertions=\"1\" name=\"$name\" time=\"$time\">
    $failure
    <system-out>
<![CDATA[
$out
]]>
    </system-out>
    </testcase>
  "
  ## testsuite block
  cat <<EOF > "$juDIR/TEST-$suite.xml"
  <testsuite failures="0" assertions="$assertions" name="$suite" tests="1" errors="$errors" time="$total">
    $content
  </testsuite>
EOF

  return $evErr
}
