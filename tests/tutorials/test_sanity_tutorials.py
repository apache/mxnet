import glob
import os
import re

# White list of non-downloadable tutorials
# Rules to be in the whitelist:
# - not a python tutorial
whitelist = ['c++/basics.md',
             'embedded/wine_detector.md',
             'r/CallbackFunction.md',
             'r/charRnnModel.md',
             'r/classifyRealImageWithPretrainedModel.md',
             'r/CustomIterator.md',
             'r/CustomLossFunction.md',
             'r/fiveMinutesNeuralNetwork.md',
             'r/index.md',
             'r/mnistCompetition.md',
             'r/ndarray.md',
             'r/symbol.md',
             'scala/char_lstm.md',
             'scala/mnist.md',
             'scala/README.md',
             'scala/mxnet_scala_on_intellij.md']
whitelist_set = set(whitelist)

def test_tutorial_downloadable():
    """
    Make sure every tutorial that isn't in the whitelist has the placeholder
    that enables notebook download
    """
    download_button_string = '<!-- INSERT SOURCE DOWNLOAD BUTTONS -->'

    tutorial_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'tutorials')
    tutorials = glob.glob(os.path.join(tutorial_path, '**', '*.md'))

    for tutorial in tutorials:
        with open(tutorial, 'r') as file:
            lines= file.readlines()
        last = lines[-1]
        second_last = lines[-2]
        downloadable = download_button_string in last or download_button_string in second_last
        friendly_name = '/'.join(tutorial.split('/')[-2:])
        if not downloadable and friendly_name  not in whitelist_set:
            print(last, second_last)
            assert False, "{} is missing <!-- INSERT SOURCE DOWNLOAD BUTTONS --> as its last line".format(friendly_name)

def test_tutorial_tested():
    """
    Make sure every tutorial that isn't in the whitelist
    has been added to the tutorial test file
    """
    tutorial_test_file = os.path.join(os.path.dirname(__file__), 'test_tutorials.py')
    f = open(tutorial_test_file, 'r')
    tutorial_test_text = '\n'.join(f.readlines())
    tutorial_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'tutorials')
    tutorials = glob.glob(os.path.join(tutorial_path, '**', '*.md'))

    tested_tutorials = set(re.findall(r"assert _test_tutorial_nb\('(.*)'\)", tutorial_test_text))
    for tutorial in tutorials:
        friendly_name = '/'.join(tutorial.split('/')[-2:]).split('.')[0]
        if friendly_name not in tested_tutorials and friendly_name+".md" not in whitelist_set:
            assert False, "{} has not been added to the tests/tutorials/test_tutorials.py test_suite".format(friendly_name)


