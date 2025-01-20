#!/usr/bin/env python
#

def a():
    from distributed_data_processing.best_movies import BestMovies

    best_movies = BestMovies(args=['-r', 'inline', 'distributed_data_processing/ratings.csv', '-o', 'distributed_data_processing/top'])
    with best_movies.make_runner() as runner:
        runner.run()
        for i, (key, value) in zip(range(8), best_movies.parse_output(runner.cat_output())):
            print(i, key, value)


def b():
    from distributed_data_processing.haters import Haters

    movie_haters = Haters(args=['-r', 'inline', 'distributed_data_processing/ratings.csv', '-o', 'distributed_data_processing/haters'])
    with movie_haters.make_runner() as runner:
        runner.run()
        # print first ten results
        for i, (key, value) in zip(range(10), movie_haters.parse_output(runner.cat_output())):
            print(i, key, value)

def c():
    from distributed_data_processing.movie_id_to_name import MovieIdToName
    joining_names = MovieIdToName(args=['-r', 'inline', 'distributed_data_processing/top', '-o', 'distributed_data_processing/top_readable'])
    with joining_names.make_runner() as runner:
        runner.run()
        print("movie name,average rating")
        for i in joining_names.parse_output(runner.cat_output()):
        	print(i[1].rstrip())


c()