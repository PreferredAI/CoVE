# Copyright 2023 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import warnings
from collections import Counter

import numpy as np

from cornac.models.recommender import NextItemRecommender


class SPop(NextItemRecommender):
    """Recommend most popular items of the current session.

    Parameters
    ----------
    name: string, default: 'SPop'
        The name of the recommender model.

    use_session_popularity: boolean, optional, default: True
        When False, no item frequency from history items in current session are being used.

    References
    ----------
    Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk:
    Session-based Recommendations with Recurrent Neural Networks, ICLR 2016
    """

    def __init__(self, name="SPop", use_session_popularity=True, use_user_popularity=False):
        """Popularity-based recommendation model.

        Args:
            G-Pop: False, False
            SG-Pop: True, False
            UG-Pop: False, True
            SUG-Pop: True, True
        """
        super().__init__(name=name, trainable=False)
        self.use_session_popularity = use_session_popularity
        self.use_user_popularity = use_user_popularity
        self.item_freq = Counter()

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        self.item_freq = Counter(self.train_set.uir_tuple[1])
        return self

    def score(self, user_idx, history_items, **kwargs):
        mode = kwargs.get("mode", "last")
        if mode is None:
            warnings.warn("No mode is provided, default to 'last'")
        # global popularity: freq/max_freq
        item_scores = np.zeros(self.total_items, dtype=np.float32)
        max_item_freq = max(self.item_freq.values()) if len(self.item_freq) > 0 else 1
        for iid, freq in self.item_freq.items():
            item_scores[iid] = freq / max_item_freq

        c_history_items = []
        if mode == "last":
            if self.use_user_popularity and self.use_session_popularity:
                c_history_items = [item for sublist in history_items for item in sublist]
            elif self.use_user_popularity:
                c_history_items = [item for sublist in history_items[:-1] for item in sublist]
            elif self.use_session_popularity:
                c_history_items = [item for sublist in history_items[-1:] for item in sublist]
        else:
            if self.use_user_popularity:
                c_history_items = [item for sublist in history_items for item in sublist]
        s_item_freq = Counter([iid for iid in c_history_items])
        for iid, cnt in s_item_freq.most_common():
            item_scores[iid] += cnt
        return item_scores
