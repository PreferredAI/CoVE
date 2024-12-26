import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from time import time

import numpy as np
import pandas as pd

from args import get_args

# file params
FILE_TYPE_PREFIX = ".hdf"

# keys
USER_KEY = "visitorid"
ITEM_KEY = "itemid"
TIME_KEY = "timestamp"
SESSION_KEY = "session_id"


SESSION_THRESHOLD = 30 * 60 * 1000
# filtering config (all methods)
MIN_ITEM_SUPPORT = 5
MIN_SESSION_LENGTH = 3
MIN_USER_SESSIONS = 3
MAX_USER_SESSIONS = None
REPEAT = False  # apply filters several times
CLEAN_TEST = True
SLICES_NUM = 5  # Preprocess the test set
SLICE_INTERVAL = 27  # total_interval = 139
DAYS_OFFSET = 0


def make_sessions(
    data, session_th=SESSION_THRESHOLD, is_ordered=False, user_key=USER_KEY, time_key=TIME_KEY, session_key=SESSION_KEY
):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:] != data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data[session_key] = session_ids
    return data


def filter_data(
    data,
    min_item_support=MIN_ITEM_SUPPORT,
    min_session_length=MIN_SESSION_LENGTH,
    min_user_sessions=MIN_USER_SESSIONS,
    max_user_sessions=MAX_USER_SESSIONS,
):
    condition = (
        data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= min_user_sessions
        and data.groupby([USER_KEY, SESSION_KEY]).size().min() >= min_session_length
        and data.groupby([ITEM_KEY]).size().min() >= min_item_support
    )
    counter = 1
    while not condition:
        print(counter)
        # keep items with >=5 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= min_item_support].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 2
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= min_session_length].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 2 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        if MAX_USER_SESSIONS is None:  # no filter for max number of sessions for each user
            good_users = sess_per_user[(sess_per_user >= min_user_sessions)].index
        else:
            good_users = sess_per_user[
                (sess_per_user >= min_user_sessions) & (sess_per_user < max_user_sessions)
            ].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = (
            data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= min_user_sessions
            and data.groupby([USER_KEY, SESSION_KEY]).size().min() >= min_session_length
            and data.groupby([ITEM_KEY]).size().min() >= min_item_support
        )
        # condition = false #if want to apply the filters once
        counter += 1
        if not REPEAT:
            break

    # output
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print(
        "Filtered data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n".format(
            len(data),
            data[USER_KEY].nunique(),
            data[SESSION_KEY].nunique(),
            data[ITEM_KEY].nunique(),
            data_start.date().isoformat(),
            data_end.date().isoformat(),
        )
    )

    return data


def split(
    data_path="processed_data/retailrocket/retail.txt",
    sep=",",
    columns=["user_id", "session_id", "item_id", "timestamp"],
    seed=123,
):
    start_time = time()
    data = pd.read_csv(data_path, header=None, names=columns, sep=sep)
    print("Data loaded in {:.2f} seconds".format(time() - start_time))
    user_sids = defaultdict(list)
    for i, row in data.iterrows():
        if row["session_id"] not in user_sids[row["user_id"]]:
            user_sids[row["user_id"]].append(row["session_id"])
    train_sids = []
    val_sids = []
    test_sids = []

    np.random.seed(seed)
    for user, sids in user_sids.items():
        if len(sids) < 3:
            train_sids.extend(sids)
            continue
        train_sids.extend(sids[:-2])
        val_test_sids = np.random.permutation(sids[-2:])
        val_sids.extend(sids[:-2])
        test_sids.extend(sids[:-2])
        val_sids.append(val_test_sids[0])
        test_sids.append(val_test_sids[1])

    train_data = data[data["session_id"].isin(train_sids)]
    val_data = data[data["session_id"].isin(val_sids)]
    test_data = data[data["session_id"].isin(test_sids)]

    # export to files
    folder_path = data_path.split("/")[:-1]
    train_data.to_csv(os.path.join(folder_path, "train.csv"), index=False, header=False)
    val_data.to_csv(os.path.join(folder_path, "val.csv"), index=False, header=False)
    test_data.to_csv(os.path.join(folder_path, "test.csv"), index=False, header=False)


if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset

    split(
        "processed_data/gowalla/check-ins.txt",
        columns=["user_id", "session_id", "item_id", "timestamp", "json"],
        sep="\t",
    )
