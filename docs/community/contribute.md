# Contributing MXNet

Apache MXNet (incubating) is a community led, open source deep learning project. We welcome new members and look forward to your contributions. Here you will find how to get started and links to detailed information on MXNet best practices and processes.


## Getting Started

The following actions are recommended steps for you to get started with contributing to MXNet.

| Action | Purpose |
|---|---|
| [Create a forum account](#forum) | Asking & answering MXNet usage questions |
| [Join the dev comm channels](#mxnet-dev-communications) | Discussions about the direction of MXNet |
| [Follow MXNet on Social Media](#social-media) | Get updates about new features and events |
| [Create a JIRA account](#jira) | Tracking tasks & prioritizing issues |
| [Check out the MXNet wiki](#confluence-wiki) | The wiki has detailed contributor information |
| [Setup MXNet for development](#setup-mxnet-for-development) | Your very own fork for creating pull requests |
| [Your first contribution](#your-first-contribution) | Complete a first contribution task |


### FAQ

* I found a bug. How do I report it?
    * [Bug submission info](#file-a-bug-report)
* I have a minor bug fix or docs update I'd like to submit. What do I do?
    * [Minor fixes process](#minor-fixes)
* I would like to submit a pull request for a significant update. What is the process?
    * [Pull request process](#formal-pull-request-process)
* I want to propose a new feature. What is the process for this?
    * [New feature process](#new-feature-process)
* What's coming next with MXNet, and how can I help?
    * [Roadmap info](#roadmap)


## MXNet Dev Communications

### Forum

If you need help with using MXNet, have questions about applying it to a particular kind of problem, or have a discussion topic, please use the discussion forum:
* [discuss.mxnet.io](https://discuss.mxnet.io) <i class="fas fa-external-link-alt"></i>

### Mailing Lists

Please join either or both of the MXNet mailing lists:

**For MXNet Users, join the USER mailing list**:

- [MXNet Apache USER mailing list](https://lists.apache.org/list.html?user@mxnet.apache.org) (user@mxnet.apache.org): To subscribe, send an email to <a href="mailto:user-subscribe@mxnet.apache.org">user-subscribe@mxnet.apache.org</a> <i class="far fa-envelope"></i>

**For Contributors to MXNet, join the DEV mailing list**:
- [MXNet Apache DEV mailing list](https://lists.apache.org/list.html?dev@mxnet.apache.org) (dev@mxnet.apache.org): To subscribe, send an email to <a href="mailto:dev-subscribe@mxnet.apache.org">dev-subscribe@mxnet.apache.org</a> <i class="far fa-envelope"></i>


* [archive](https://lists.apache.org/list.html?dev@mxnet.apache.org) <i class="fas fa-external-link-alt"></i>

### Slack

To join the MXNet slack channel send request to the contributor mailing list.
 * <a href="mailto:dev@mxnet.apache.org?subject=Requesting%20slack%20access">email</a> <i class="far fa-envelope"></i>
 * [archive](https://the-asf.slackarchive.io/mxnet) <i class="fas fa-external-link-alt"></i>


### Social Media

Keep connected with the latest MXNet news and updates on [Twitter](https://twitter.com/apachemxnet) and [Reddit](https://reddit.com/r/mxnet). Also, subscribe to the [MXNet YouTube channel](https://www.youtube.com/channel/UCQua2ZAkbr_Shsgfk1LCy6A).

<div class="g-ytsubscribe" data-channelid="UCQua2ZAkbr_Shsgfk1LCy6A" data-layout="full" data-count="hidden"></div>
<br/><br/>
<a href="https://twitter.com/apachemxnet?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="false">Follow @apachemxnet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
<br/><br/>
<a href="https://reddit.com/r/mxnet"><img src="https://www.redditstatic.com/spreddit5.gif" alt="reddit" border="0"/> r/mxnet</a>


## JIRA

MXNet uses Apache's JIRA to track issues and larger projects. Anyone can review open issues, but in order create issues or view JIRA boards, you must create an account.

* [Open JIRA Issues](https://issues.apache.org/jira/projects/MXNET/issues)
* [JIRA boards](https://issues.apache.org/jira/secure/RapidBoard.jspa) <i class="fas fa-lock"></i>


## Confluence Wiki

The [MXNet Confluence Wiki](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Home) has detailed development environment setup info, design proposals, release process info, and more. This is generally where contributor information is maintained.

* [MXNet Confluence Wiki](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Home) <i class="fas fa-external-link-alt"></i>


## Setup MXNet for Development

The process for setting up MXNet for development depends on several factors, and is constantly being improved and expanded for more development languages. Setup information is on the MXNet Confluence Wiki.

* [MXNet Confluence Wiki: Development](https://cwiki.apache.org/confluence/display/MXNET/Development) <i class="fas fa-external-link-alt"></i>


## Your First Contribution

**Step 1**: Visit the project on GitHub and review the [calls for contribution](https://github.com/apache/incubator-mxnet/labels/Call%20for%20Contribution). Click the GitHub button:
<a class="github-button" href="https://github.com/apache/incubator-mxnet/labels/Call%20for%20Contribution" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a>

**Step 2**: Tackle a smaller issue or improve documentation to get familiar with the process. As part of your pull request, add your name to [CONTRIBUTORS.md](https://github.com/apache/incubator-mxnet/blob/master/CONTRIBUTORS.md).

**Step 3**: Follow the [formal pull request (PR) process](#formal-pull-request-process) to submit your PR for review.

**Important**: keep an eye on your pull request, respond to comments and change requests, and rebase or resubmit your PR if fails the Jenkins continuous integration tests. Ask for help in the [forum or slack channel](#mxnet-dev-communications) if you get stuck.


## File a bug report

Please let us know if you experienced a problem with MXNet. Please provide detailed information about the problem you encountered and, if possible, add a description that helps to reproduce the problem. You have two alternatives for filing a bug report:
<p><a href="http://issues.apache.org/jira/browse/MXNet"><i class="fas fa-bug"></i> JIRA</a></p>
<p><a href="https://github.com/apache/incubator-mxnet/issues"><i class="fab fa-github"></i> GitHub</a></p>


## Minor Fixes

If you have found an issue and would like to contribute a bug fix or documentation update, follow these guidelines:

* If it is trivial, just create a [pull request](https://github.com/apache/incubator-mxnet/pulls).
* If it is non-trivial, you should follow the [formal pull request process](#formal-pull-request-process) described in the next section.


## Formal Pull Request Process

Any new features of improvements that are non-trivial should follow the complete flow of:

1. [Review the contribution standards](https://cwiki.apache.org/confluence/display/MXNET/Development+Process) for your type of submission.
1. [Create a JIRA issue](https://issues.apache.org/jira/secure/CreateIssue!default.jspa).
1. [Create the PR on GitHub](https://github.com/apache/incubator-mxnet/pulls) and add the JIRA issue ID to the PR's title.

Further details on this process can be found on the [Wiki](https://cwiki.apache.org/confluence/display/MXNET/Development).


## New Feature Process

Our community is constantly looking for feedback to improve Apache MXNet. If you have an idea how to improve MXNet or have a new feature in mind that would be beneficial for MXNet users, please open an issue in [MXNet’s JIRA](http://issues.apache.org/jira/browse/MXNet). The improvement or new feature should be described in appropriate detail and include the scope and its requirements if possible. Detailed information is important for a few reasons:<br/>
- It ensures your requirements are met when the improvement or feature is implemented.<br/>
- It helps to estimate the effort and to design a solution that addresses your needs. <br/>
- It allows for constructive discussions that might arise around this issue.

Detailed information is also required, if you plan to contribute the improvement or feature you proposed yourself. Please read the [contributions](http://mxnet.io/community/contribute.html) guide in this case as well.


## Roadmap

MXNet is evolving fast. To see what's next and what the community is currently working on, check out the Roadmap issues on GitHub and the JIRA Boards:

<a class="github-button" href="https://github.com/apache/incubator-mxnet/labels/Roadmap" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Roadmap</a>
<br/>
[JIRA boards](https://issues.apache.org/jira/secure/RapidBoard.jspa) <i class="fas fa-lock"></i>


<script defer src="https://use.fontawesome.com/releases/v5.0.12/js/all.js" integrity="sha384-Voup2lBiiyZYkRto2XWqbzxHXwzcm4A5RfdfG6466bu5LqjwwrjXCMBQBLMWh7qR" crossorigin="anonymous"></script>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>


## Contributors
MXNet has been developed by and is used by a group of active community members. Contribute to improving it!

<i class="fab fa-github"></i> [Contributors and Committers](https://github.com/apache/incubator-mxnet/blob/master/CONTRIBUTORS.md)
