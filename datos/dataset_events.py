import numpy as np
import fileinput
import pickle


def get_yahoo_events(filenames):
    events = []
    # Remover los outliers que no tienen features
    outlier = '109528'
    int_outlier = 109528
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
                # Remover los outliers que no tienen features
                if np.any(opciones == int_outlier):
                    opciones = np.delete(opciones, np.where(opciones == int_outlier))
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