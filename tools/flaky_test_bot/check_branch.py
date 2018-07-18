import argparse
import flakiness_checker
import diff_collator

if __name__ == "__main__":
    args = diff_collator.parse_args()
    diff_output = diff_collator.get_diff_output(args)
    changes = diff_collator.parser(diff_output)
    diff_collator.output_changes(changes, args.verbosity)


