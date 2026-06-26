import pickle

def save_var(var, filename, do_print = True):
    '''Saving the objects:'''
    try:
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(var, f)
    except Exception as e:
        do_print and print('Could not save', filename)
        return False
    
    return True


def load_var(filename, do_print = True, use_latin1 = False):
    '''Getting back the objects:'''
    try:
        with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f) if not use_latin1 else pickle.load(f, encoding='latin1')
    except Exception as e:
        do_print and print('Could not load', filename, 'because', e)
        return None, False

    return var, True