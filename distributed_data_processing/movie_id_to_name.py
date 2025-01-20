import csv
import io

from mrjob.job import MRJob
from mrjob.protocol import TextValueProtocol


# "map side join", applicable when the file to be joined with is small, such as this movies.csv file
# alternative is to run specialized MR join using both files as mapper input, grouping by keys, and effectively joining in the reducer, thus a "reducer side join"
class MovieIdToName(MRJob):

    #
    def files(self):
        return ['distributed_data_processing/movies.csv']

    def mapper_init(self):
        lookup = {}
        with open('movies.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                lookup[row[0]] = row[1]
        self.lookup = lookup

    def mapper(self, _, line):
        parts = line.split(',')
        movie_id = parts.pop(0)
        # properly encode as csv to handle commas in movie names by quoting
        with io.StringIO() as s:
            csv.writer(s).writerow([self.lookup[movie_id]] + parts)
            yield None, s.getvalue().strip('\r\n')

    OUTPUT_PROTOCOL = TextValueProtocol


if __name__ == '__main__':
    MovieIdToName.run()
