"""
Line format for yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000

Some log files contain rows with erroneous data.

After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""

import numpy as np
import fileinput
import pickle


def get_yahoo_events(filenames):
    """
    Reads a stream of events from the list of given files.
    
    Parameters
    ----------
    filenames : list
        List of filenames
    
    Stores
    -------    
    articles : [article_ids]
    features : [[article_1_features] .. [article_n_features]]
      : [
                 0 : displayed_article_index (relative to the pool),
                 1 : user_click,
                 2 : [user_features],
                 3 : [pool_indexes]
             ]
    """
    events = []
    # Remover los outliers que no tienen features
    outlier = '109528'
    # Total 4681993
    breakThresshold = 990000
    batch_size = 100000
    b_num = 0

    with open('users.pkl', 'rb') as fp:
        users = pickle.load(fp)

    user_ids = [u['id'] for u in users]

    with fileinput.input(files=filenames) as f:
        i = 0
        for line in f:
            cols = line.split('|')

            user_feature = cols[1].split()[1:6]
            user_feature = np.array([float(f[2:]) for f in user_feature]).reshape((len(user_feature), 1))

            user_id = 0
            for j in range(len(user_feature)):
                user_id += int(user_feature[j] * 1000000) ** j

            article = (cols[0].split()[1])

            if (user_ids is not None and user_id in user_ids) and article != outlier:
                i += 1

                article = int(article)

                opciones = np.zeros(len(cols) - 2, dtype=int)
                for h in range(2, len(cols)):
                    opciones[h - 2] = cols[h].split()[0]
                events.append(
                    (article, cols[0].split()[2] == '1', opciones, user_id))  # art_id:Click:UserFeature))

            if i == batch_size:
                i = 0
                b_num += 1
                print(b_num)
                with open('events' + str(b_num) + '.pkl', 'wb') as fp:
                    pickle.dump(events, fp)
                    del events
                    events = []

            if breakThresshold is not None and b_num * batch_size + i == breakThresshold:
                break

    print(b_num * batch_size + i)
    if len(events) != 0:
        with open('events' + str(b_num + 1) + '.pkl', 'wb') as fp:
            #print(events)
            pickle.dump(events, fp)

    # Tests


if __name__ == '__main__':
    files = ("ydata-fp-td-clicks-v1_0.20090501")

    get_yahoo_events(files)