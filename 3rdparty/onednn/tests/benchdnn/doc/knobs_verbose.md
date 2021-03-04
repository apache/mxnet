# Verbose

**Benchdnn** provides execution information depending on verbosity level. The
level setting is controlled by the `-vN` or `--verbose=N` option. The higher the
level, the more information is provided. Each subsequent level appends new
information in addition to the previous level to be printed. Currently, the
following information is printed for certain verbosity levels:
* Level 0 (the default).
    * Various errors.
    * Execution statistics.
    * When the status of the problem is `FAILED`, prints a short version of
      comparison summary.
    * Performance template used.
* Level 1.
    * Problem reproducer line right after the problem was constructed. It is
      convenient to catch the repro line in case of a program crash.
* Level 2.
    * Various warnings.
    * The library implementation name if it hits the --skip-impl match.
* Level 5.
    * The library implementation name picked to compute the given problem.
* Level 50.
    * Full path of batch file used.
* Level 99.
    * A full version of comparison summary.

