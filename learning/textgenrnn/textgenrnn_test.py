from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.generate()

textgen.train_from_file('/Users/codelife/Developer/tensorFlow/DeepLearningZeroToAll/textgenrnn/nate_news.txt', num_epochs=1)
textgen.generate()

