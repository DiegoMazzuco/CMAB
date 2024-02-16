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
    articles = {}
    users = {}
    # Remover los outliers que no tienen features
    outlier = '109528'
    # Total 4681993
    # Probabilidad 4.07%
    breakThresshold = None
    b_num = 0
    click_amount = 0

    with fileinput.input(files=filenames) as f:
        i = 0
        for line in f:
            cols = line.split('|')

            user_feature = cols[1].split()[1:6]
            user_feature = [float(f[2:]) for f in user_feature]

            user_id = 0
            for j in range(len(user_feature)):
                user_id += int(user_feature[j] * 1000000) ** j

            if users.get(user_id) is None:
                users[user_id] = {
                    'clicks': 0,
                    'views': 0,
                    'features': user_feature,
                    'id': user_id
                }

            article = (cols[0].split()[1])

            if article != outlier:
                i += 1

                feature = [f for f in cols if article in f][1].split()[1:6]
                feature = [float(f[2:]) for f in feature]

                article = int(article)

                if articles.get(article) is None:
                    articles[article] = {
                        'clicks': 0,
                        'views': 0,
                        'features': feature,
                        'id': article
                    }

                click = cols[0].split()[2] == '1'

                if click:
                    #user
                    print(i)
                    click_amount += 1
                    users[user_id]['clicks'] += 1
                    #article
                    print(i)
                    click_amount += 1
                    articles[article]['clicks'] += 1
                articles[article]['views'] += 1
                users[user_id]['views'] += 1

                if breakThresshold is not None and i == breakThresshold:
                    break

    for article in articles:
        articles[article]['probability'] = articles[article]['clicks'] / articles[article]['views']

    print('Probabilidad Random')
    print(click_amount*100/i)

    with open('articles.pkl', 'wb') as fp:
        pickle.dump(articles, fp)

    with open('users.pkl', 'wb') as fp:
        pickle.dump(list(users.values()), fp)
    print('finished')

    # Tests


if __name__ == '__main__':
    files = ("ydata-fp-td-clicks-v1_0.20090501")

    get_yahoo_events(files)

    with open('articles.pkl', 'rb') as fp:
        articles = pickle.load(fp)
        print('Articles dictionary')
        print(articles)

    with open('users.pkl', 'rb') as fp:
        users = pickle.load(fp)
        print('Users array')
        print(users)
