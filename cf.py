#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from tqdm import tqdm


class CF():
    """Item-based Nearest Neighbor Collaborative Filtering model.

    Parameters
    ----------

    n_neighbors : int, default=20
        Number of nearest neighbors to use when predicting rating.

    metric : {'pearson', 'spearman', 'cosine', 'mse'}, default="cosine"
        Method of calculating similarity between movies.

    user_cutoff : int, default=None
        Minimum number of ratings per user. Users with fewer ratings
        are removed from the training data.

    movie_cutoff : int, default=100
        Minimum number of ratings per movie. Movies with fewer ratings
        are removed from the training data.

    Attributes
    ----------

    _ratings : pd.Dataframe
        Filtered training data stored as a pandas DataFrame with MultiIndex.

    _similarities : pd.DataFrame
        Similarity matrix for every pair of movies.

    _mean_user_ratings : pd.Series
        Mean rating of every user.

    _mean_movie_ratings : pd.Series
        Mean rating of every movie.

    _global_ratings_mean : float
        Mean rating of all movies in the database.

    _METRICS : Dict[str, callable]
        Mapping from string indicator of a metric to its corresponding method.

    """

    def __init__(self, n_neighbors=20, metric="cosine", user_cutoff=None, movie_cutoff=100):
        self._ratings = None
        self._similarities = None
        self._mean_user_ratings = None
        self._mean_movie_ratings = None
        self._global_ratings_mean = None
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.user_cutoff = user_cutoff
        self.movie_cutoff = movie_cutoff

        self._METRICS = {
            "pearson" : self._pearson,
            "spearman": self._spearman,
            "cosine"  : self._cosine,
            "mse"     : self._mse,
        }


    def _filter_infrequent(self, data, col, cutoff):
        """Remove rows with infrequent values.

        All rows in `data` that contain a value in the column `col`
        with fewer than `cutoff` occurences are removed.

        Parameters
        ----------
        data : pd.DataFrame
            User ratings.

        col : str

        cutoff : int

        Returns
        -------
        data : pd.DataFrame
            The filtered DataFrame.
        """

        if cutoff is None:
            return data

        counts = data[col].value_counts()
        infrequent = counts[counts < cutoff].index
        return data[~data[col].isin(infrequent)]


    def fit(self, X):
        """Store data and precompute statistics.

        Store user ratings in a MultiIndex DataFrame, calculate mean ratings
        for each user and each movie, calculate the mean of all ratings, and
        precompute a similarity matrix for each pair of movies.

        Parameters
        ----------
        X : pd.DataFrame
            User ratings.
        """

        X = X.drop(columns="timestamp")

        X = self._filter_infrequent(X, "userId", self.user_cutoff)
        X = self._filter_infrequent(X, "movieId", self.movie_cutoff)

        X = X.set_index(["movieId", "userId"])
        self._ratings = X.copy()

        self._mean_user_ratings = self._ratings.rating.mean(level="userId")
        self._mean_movie_ratings = self._ratings.rating.mean(level="movieId")
        self._global_ratings_mean = self._ratings.rating.values.mean()

        # create similarity matrix of shape (n_movies, n_movies)
        movies = self._ratings.index.levels[0]
        self._similarities = pd.DataFrame(
            data    = np.zeros((movies.size, movies.size)),
            index   = movies,
            columns = movies,
        )

        # fill the matrix
        for m1 in tqdm(movies):
            for m2 in movies:
                if m1 == m2:
                    self._similarities.loc[m1, m1] = 1.0
                elif m1 < m2:
                    sim = self._similarity(m1, m2)
                    self._similarities.loc[m1, m2] = sim
                    self._similarities.loc[m2, m1] = sim


    def _pearson(self, x, y):
        """Calculate similarity based on Pearson's correlation coefficient.

        Parameters
        ----------
        x : ndarray
            Vector of ratings of movie 1.

        y : ndarray
            Vector of ratings of movie 2.

        Returns
        -------
        float
            The calculated Pearson similarity.
        """

        # constant arrays result in division by 0
        if np.all(x == x.values[0]) or np.all(y == y.values[0]):
            return 0.0

        return (pearsonr(x, y)[0] + 1) / 2


    def _spearman(self, x, y):
        """Calculate similarity based on Spearman's rank correlation coefficient.

        Parameters
        ----------
        x : ndarray
            Vector of ratings of movie 1.

        y : ndarray
            Vector of ratings of movie 2.

        Returns
        -------
        float
            The calculated Spearman similarity.
        """

        # constant arrays result in division by 0
        if np.all(x == x.values[0]) or np.all(y == y.values[0]):
            return 0.0

        return (spearmanr(x, y)[0] + 1) / 2


    def _cosine(self, x, y):
        """Calculate adjusted cosine similarity.

        Uses scipy.spatial.distance.cosine to calculate cosine distance which
        is a number between -1 and 1. The similarity is calculated as 1-distance
        and further scaled into an interval 0 to 1.

        Parameters
        ----------
        x : ndarray
            Vector of ratings of movie 1.

        y : ndarray
            Vector of ratings of movie 2.

        Returns
        -------
        float
            The calculated cosine similarity.

        """

        # array of zeros results in division by 0
        if np.all(x == 0) or np.all(y == 0):
            return 0.0

        return ( 2 - cosine(x, y) ) / 2


    def _mse(self, x, y):
        """Calculate mean square error similarity.

        Similarity = 1 / mse.

        Parameters
        ----------
        x : ndarray
            Vector of ratings of movie 1.

        y : ndarray
            Vector of ratings of movie 2.

        Returns
        -------
        float
            The calculated MSE similarity.
        """

        return 1 / ( 1 + np.square(x - y).mean() )


    def _jaccard_index(self, m1, m2):
        """Calculate Jaccard Index between two movies.

        Parameters
        ----------
        m1 : pd.Index
            Set of users that rated movie 1.

        m2 : pd.Index
            Set of users that rated movie 2.

        Returns
        -------
        float
            The calculated Jaccard index.
        """

        a = m1.size
        b = m2.size
        a_b = np.intersect1d(m1, m2).size

        return a_b / (a + b - a_b)


    def _similarity(self, movie_id_1, movie_id_2):
        """Calculate similarity between two movies.

        Uses `self.metric` and weighs by the jaccard index.

        Parameters
        ----------
        movie_id_1 : int

        movie_id_2 : int

        Returns
        -------
        float
            The calculated similarity.
        """

        # get subsets of ratings that correspond to the movies of interest
        m1 = self._ratings.xs(movie_id_1, level="movieId")
        m2 = self._ratings.xs(movie_id_2, level="movieId")

        # ratings of users that rated both movies
        merge = pd.merge(m1, m2, on="userId")

        if merge.size == 0:
            return 0.0

        x, y = merge.rating_x, merge.rating_y

        # adjust for user bias
        biases = self._mean_user_ratings.loc[merge.index]
        x -= biases
        y -= biases

        metric = self._METRICS[self.metric]

        jaccard = self._jaccard_index(m1.index, m2.index)

        result = metric(x, y) * jaccard

        if not np.isfinite(result):
            return 0.0

        return result


    def _rate(self, mean_rating, ratings, weights, biases):
        """Calculate rating of a movie based on ratings of similar movies.

        Parameters
        ----------
        mean_rating : float
            Mean rating of the movie whose user rating is being predicted.

        ratings : ndarray
            Ratings of movies the user has seen.

        weights : ndarray
            Similarities of movies the user has seen to the movie we are
            predicting the rating of.

        biases : ndarray
            Mean ratings of the movies the user has seen.

        Returns
        -------
        float
            Predicted rating.
        """

        # `self.n_neighbors` largest weights are kept, others are set to zero
        weights *= np.argsort(weights) < self.n_neighbors

        if weights.sum() == 0:
            return mean_rating

        # how much better or worse has the user rated a movie compared
        # to that movie's mean rating
        ratings -= biases

        return mean_rating + np.average(ratings, weights=weights)


    def _predict_rating(self, user_df, movie):
        """Predict rating of `movie` based on user history.

        Parameters
        ----------
        user_df : pd.DataFrame
            User ratings.

        movie : int
            Movie ID.

        Returns
        -------
        float
            Predicted rating.
        """
        if movie not in self._mean_movie_ratings.index:
            return self._global_ratings_mean

        # filter movies that are not in the database
        user_df = user_df.loc[
            user_df.index.intersection(self._mean_movie_ratings.index)
        ]

        if user_df.size == 0:
            return self._mean_movie_ratings[movie]

        weights = self._similarities.loc[user_df.index, movie]
        biases = self._mean_movie_ratings.loc[user_df.index]

        return self._rate(
            self._mean_movie_ratings[movie],
            user_df.values,
            weights,
            biases
        )


    def predict(self, X):
        """Predict user ratings.

        For every user, incrementaly take their movie ratings and predict
        the rating of the next movie.

        Parameters
        ----------
        X : pd.DataFrame
            User ratings.

        Returns
        -------
        predictions : ndarray
            Predicted ratings.
        """

        predictions = np.zeros((X.index.size))
        idx = 0

        X = X.set_index(["userId", "movieId"])

        for user in X.index.levels[0]:
            if user in self._ratings.index.levels[1]:
                user_ratings = self._ratings.xs(user, level="userId").rating
            else:
                user_ratings = pd.Series(dtype=np.float64)

            user_new = X.xs(user, level="userId").rating

            for i in range(user_new.size):
                ratings = pd.concat([user_ratings, user_new[:i]])
                movie = user_new.index[i]
                predictions[idx] = self._predict_rating(ratings, movie)
                idx += 1

        return predictions


    def recommend(self, user_df, k):
        """Recommend `k` movies based on user's history.

        Parameters
        ----------
        user_df : pd.DataFrame
            User ratings.

        k : int
            How many movies will be recommended.

        Returns
        -------
        ndarray
            IDs of the recommended movies.
        """

        # filter movies that are not in the database
        user_df = user_df.loc[
            user_df.index.intersection(self._mean_movie_ratings.index)
        ]

        # recommend top rated movies if empty
        if user_df.size == 0:
            return self._mean_movie_ratings.nlargest(k).index.values

        # disregard movies the user has already seen
        rows = self._similarities.index.difference(user_df.index)

        # calculate score for each movie as mean similarity to all movies
        # the user has seen weighed by rating
        scores = np.mean(
            self._similarities.loc[rows, user_df.index] * user_df.rating,
            axis=1
        )

        return scores.nlargest(k).index.values
