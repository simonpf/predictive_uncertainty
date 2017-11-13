# Inline matplotlib
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("matplotlib inline")

# Matplotlib style
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams["figure.figsize"] = (8,6)
matplotlib.rcParams["font.size"] = 14

globals()["plt"] = plt
globals()["matplotlib"] = matplotlib
