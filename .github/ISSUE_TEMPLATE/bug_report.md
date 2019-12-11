---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'Bug'
assignees: ''

---
## Description
(A clear and concise description of what the bug is.)

### Error Message
(Paste the complete error message. Please also include stack trace by setting environment variable `DMLC_LOG_STACK_TRACE_DEPTH=10` before running your script.)

## To Reproduce
(If you developed your own code, please provide a short script that reproduces the error. For existing examples, please provide link.)

### Steps to reproduce
(Paste the commands you ran that produced the error.)

1.
2.

## What have you tried to solve it?

1.
2.

## Environment

We recommend using our script for collecting the diagnositc information. Run the following command and paste the outputs below:
```
curl --retry 10 -s https://raw.githubusercontent.com/dmlc/gluon-nlp/master/tools/diagnose.py | python

# paste outputs here
```
