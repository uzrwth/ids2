import mrjob
from collections.abc import Iterator, Iterable
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import TextValueProtocol, PickleValueProtocol


class BestMovies(MRJob):
    # this pre-processing mapper is for convenience
    # it unpacks the string value into the logical signature of the file
    def map_pre(self, _: None, line: str) -> Iterator[tuple[int, tuple[str, str, float, str, float]]]:
        i, user_id, movie_id, rating, timestamp, rating_normalized = line.split(',')
        yield i, (user_id, movie_id, float(rating), timestamp, float(rating_normalized))

    def map_1(self, _: int, line: tuple[str, str, float, str, float]):
        user_id, movie_id, rating, timestamp, rating_normalized = line
        yield movie_id, (1, rating)

    def reduce_1(self, movie_id, values):
    	total_reviews = 0
    	total_ratings = 0
    	for (reviews, rating) in values:
    		total_reviews += reviews
    		total_ratings += rating
    	avg = total_ratings/total_reviews
    	if avg >= 4 and total_reviews >= 10:
    		yield movie_id, avg

    def map_2(self, movie_id, avg):
    	yield None, (avg, movie_id)

    def reduce_2(self, _, values):
    	topten = []
    	for p in values:
    		topten.append(p)
    		topten.sort()
    		topten = topten[-8:]

    	for (avg, movie_id) in topten:
    		yield None, (movie_id, avg)


    # this post-processing mapper is for convenience
    # the output from the last reducer step is simply written as text into a file, ignoring the key
    def map_post(self, _, pair: tuple[str, float]):
        yield None, ','.join(map(str, pair))

    # the output from the last reducer step is simply written as text into a file, ignoring the key
    OUTPUT_PROTOCOL = TextValueProtocol

    # this can be treated as boilerplate
    def steps(self):
        return [MRStep(mapper=self.map_pre),
                MRStep(mapper=self.map_1, reducer=self.reduce_1),
                MRStep(mapper=self.map_2, reducer=self.reduce_2),
                MRStep(mapper=self.map_post)]


if __name__ == '__main__':
    BestMovies.run()


