import mrjob
from collections.abc import Iterator, Iterable
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import TextValueProtocol


class Haters(MRJob):


    def mapper(self, _, line: str):
    	i, user_id, movie_id, rating, timestamp, rating_normalized = line.split(',')
    	if float(rating) < 2:
    		yield user_id, 1

    def combiner(self, key, values):
    	yield key, sum(values)


    def reducer(self, key, values):
    	reviews = sum(values)
    	if reviews >= 50:
    		yield None, key


if __name__ == '__main__':
    Haters.run()


