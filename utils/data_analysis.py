###
#
# Dataset Analysis Functions
#
###
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def word_zipf_distribution(vocab_file, a=1.5, verbose=False, incl_stop=True, log_scale=True):
    """
    Plot the (sorted) word frequencies and
    fit the plot with a Zipf distribution
    
    Requirements
    ------------
    import csv

    Parameters
    ----------
    vocab_file : str
        path to the vocabulary file
    a : float, optional
        Zipf distribution parameter
        (default: 1.5)
    verbose : bool, optional
        whether to print out information
        about the document (default: False)
    incl_stop : bool, optional
        whether to include stop words in
        the plots (default: True)
    log_scale : bool, optional
        whether to use log-log scale
        (default: True)
    
    Notes
    -----
    The vocabulary file is assumed to be a
    CSV file with the following columns:
    - 0 : word
    - 1 : counts
    - 2 : frequency (counts/num. words)
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        
        i = 0
        total = 0.
        
        freqs = []
        freqs_no_stops = []
        labels = []
        labels_no_stops = []
        
        current = 1.
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        
        for row in data:
            i += 1
            freqs.append(float(row[2]))
            labels.append(row[0])
            
            if row[0] not in stopwords:
                freqs_no_stops.append(float(row[2]))
                labels_no_stops.append(row[0])
            else:
                if verbose:
                    print('STOP: ', row[0])
        
        if incl_stop:
            freqs = freqs_no_stops
            labels = labels_no_stops
        
        x = np.arange(len(freqs))
        
        plt.plot(x, freqs, 'b+-')
        plt.xticks(x, labels, rotation='vertical')
        
        # Zipf distribution
        x_zipf = x + 1
        y = x_zipf**(-a) / special.zetac(a)
        plt.plot(x, y/max(y), linewidth=2, color='r')
        
        if verbose:
            print(i, ' rows')
            print('freqs len: ', len(freqs))
            print('MAX Y ', max(y))
        
        plt.show()