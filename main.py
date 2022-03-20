import random

import numpy
import numpy as np


class Candidate:
    def __init__(self, immutable_squares, config):
        self.immutable_squares = immutable_squares
        self.config = config

    def get_config(self):
        return self.config

    # fitness functions
    def determine_cols_fitness(self):
        columns_fitness = 0
        for i in range(9):
            column = self.config[:, i]
            unique_vals = len(np.unique(column))
            columns_fitness += 1 / (9 * (10 - unique_vals))
        return columns_fitness

    def determine_rows_fitness(self):
        rows_fitness = 0
        for i in range(9):
            row = self.config[i, :]
            unique_vals = len(np.unique(row))
            rows_fitness += 1 / (9 * (10 - unique_vals))
        return rows_fitness

    def determine_squares_fitness(self):
        squares_fitness = 0
        for i in range(3):
            for j in range(3):
                square = self.config[i * 3:i * 3 + 3, j * 3:j * 3 + 3]
                square_as_list = square.flatten()
                unique_vals = len(np.unique(square_as_list))
                squares_fitness += 1 / (9 * (10 - unique_vals))
        return squares_fitness

    def determine_fitness(self):
        fitness = (self.determine_cols_fitness() + self.determine_cols_fitness() + self.determine_squares_fitness()) / 3
        return fitness


class Solver:
    def __init__(self, input):
        self.input = input

    def get_immutable_squares_coordinates(self):
        index = 0

        immutable_squares_coordinates = {}
        # iterate over rows
        for i in range(9):
            # iterate over values in rows
            for j in range(9):
                if self.input[index] != '0':
                    immutable_squares_coordinates[(j, i)] = self.input[index]
                index += 1
        return immutable_squares_coordinates

    def fill_immutable_squares(self, config):
        immutable_squares = self.get_immutable_squares_coordinates()
        legal_config = config
        for i in immutable_squares:
            x, y = i
            legal_config[y][x] = immutable_squares[i]
        return legal_config

    def create_candidate(self):
        config = np.random.randint(1, 9, size=(9, 9))
        config = self.fill_immutable_squares(config)
        return Candidate(self.get_immutable_squares_coordinates(), config)

    def mutate(self, candidate, mutation_rate):
        r = random.uniform(0, 1)
        if r < mutation_rate:
            config = np.copy(candidate.config)
            # choose x and y coordinates of square a
            ax = random.randint(0, 8)
            ay = random.randint(0, 8)

            # choose x and y coordinates of square b
            bx = random.randint(0, 8)
            by = random.randint(0, 8)

            # pick points in the matrix
            a = config[ay][ax]
            b = config[by][bx]

            # swap squares a and b
            config[ay][ax] = b
            config[by][bx] = a
            return config
        else:
            return np.copy(candidate.config)

    def crossover_columns(self, candidate_a, candidate_b):
        # the column chosen from candidate a
        column_a_number = random.randint(0, 8)
        column_a = np.copy(candidate_a.config[:, column_a_number])

        # the column chosen from candidate b
        column_b_number = random.randint(0, 8)
        column_b = np.copy(candidate_b.config[:, column_b_number])

        # the number of cells chosen from column a
        div_point = random.randint(0, 8)

        new_column_a = numpy.concatenate([column_a[:div_point], column_b[div_point:]])
        new_column_b = numpy.concatenate([column_b[:div_point], column_a[div_point:]])

        # create configs for new candidates c and d
        candidate_c_config = np.copy(candidate_a.config)
        candidate_c_config[:, column_a_number] = new_column_a

        candidate_d_config = np.copy(candidate_b.config)

        candidate_d_config[:, column_b_number] = new_column_b

        candidate_c = Candidate(self.get_immutable_squares_coordinates(), candidate_c_config)
        candidate_d = Candidate(self.get_immutable_squares_coordinates(), candidate_d_config)

        return candidate_c, candidate_d

    def crossover_rows(self, candidate_a, candidate_b):
        # the row chosen from candidate a
        row_a_number = random.randint(0, 8)
        row_a = candidate_a.config[row_a_number]

        # the row chosen from candidate b
        row_b_number = random.randint(0, 8)
        row_b = candidate_b.config[row_b_number]

        # the number of cells chosen from row a
        div_point = random.randint(0, 8)

        new_row_a = numpy.concatenate([row_a[:div_point], row_b[div_point:]])
        new_row_b = numpy.concatenate([row_b[:div_point], row_a[div_point:]])

        # create configs for new candidates c and d
        candidate_c_config = np.copy(candidate_a.config)
        candidate_c_config[row_a_number] = new_row_a

        candidate_d_config = np.copy(candidate_b.config)
        candidate_d_config[:, row_b_number] = new_row_b

        candidate_c = Candidate(self.get_immutable_squares_coordinates(), candidate_c_config)
        candidate_d = Candidate(self.get_immutable_squares_coordinates(), candidate_d_config)

        return candidate_c, candidate_d

    def crossover_squares(self, candidate_a, candidate_b):
        candidate_a_squares = []
        candidate_b_squares = []

        # get a list of squares as lists from candidate a
        for i in range(3):
            for j in range(3):
                square = candidate_a.config[i * 3:i * 3 + 3, j * 3:j * 3 + 3]
                square_as_list = square.flatten()
                candidate_a_squares.append(square_as_list)

        # get a list of squares as lists from candidate b
        for i in range(3):
            for j in range(3):
                square = candidate_b.config[i * 3:i * 3 + 3, j * 3:j * 3 + 3]
                square_as_list = square.flatten()
                candidate_b_squares.append(square_as_list)

        # the square chosen from candidate a
        square_a_number = random.randint(0, 8)
        square_a = np.copy(candidate_a.config[square_a_number])

        # the square chosen from candidate b
        square_b_number = random.randint(0, 8)
        square_b = np.copy(candidate_b.config[square_b_number])

        # the number of cells chosen from row a
        div_point = random.randint(0, 8)

        new_square_a = numpy.concatenate([square_a[:div_point], square_b[div_point:]])
        new_square_a = new_square_a.reshape(3, 3)

        new_square_b = numpy.concatenate([square_b[:div_point], square_a[div_point:]])
        new_square_b = new_square_b.reshape(3, 3)

        # convert square a number into a position on the grid
        square_a_number = random.randint(0,8)
        square_a_x = 0
        square_a_y = 0
        count = 0
        for i in range(3):
            for j in range(3):
                count += 1
                square_a_x = j
                square_a_y = i
                if count > square_a_number:
                    break
            else:
                continue
            break

        # convert square b number into a position on the grid
        square_b_number = random.randint(0, 8)
        square_b_x = 0
        square_b_y = 0
        count = 0
        for i in range(3):
            for j in range(3):
                count += 1
                square_b_x = j
                square_b_y = i
                if count > square_a_number:
                    break
            else:
                continue
            break

        # create configs for new candidates c and d
        candidate_c_config = np.copy(candidate_a.config)
        candidate_c_config[square_a_y * 3:square_a_y * 3 + 3, square_a_x * 3:square_a_x * 3 + 3]\
            = new_square_a

        candidate_d_config = np.copy(candidate_b.config)
        candidate_d_config[square_b_y * 3:square_b_y * 3 + 3, square_b_x * 3:square_b_x * 3 + 3] \
            = new_square_b

        candidate_c = Candidate(self.get_immutable_squares_coordinates(), candidate_c_config)
        candidate_d = Candidate(self.get_immutable_squares_coordinates(), candidate_d_config)

        return candidate_c, candidate_d

    def generate_random_population(self, population_size):
        population = []
        for i in range(population_size):
            population.append(self.create_candidate())
        population.sort(key=lambda x: x.determine_fitness(), reverse=False)
        return population

    def solve(self, population_size, num_of_gens, elite_threshold, selection_rate, mutation_rate):
        # generate an initial population
        pop = self.generate_random_population(population_size)

        for gen in range(0, num_of_gens):
            print(pop[0].config)
            print('generation', gen)
            print('fitness=', pop[0].determine_fitness())
            print('row fitness=', pop[0].determine_rows_fitness())
            print('column fitness=', pop[0].determine_cols_fitness())
            print('squares fitness=', pop[0].determine_squares_fitness())

            next_gen = []
            num_of_elite = round(population_size * elite_threshold)

            # select elite candidates to automatically be selected
            for e in range(0, num_of_elite):
                next_gen.append(pop[e])

            for count in range(num_of_elite, population_size, 2):

                # # select crossover function
                # f = random.randint(0, 1)
                # child_a = None
                # child_b = None
                # if f == 0:
                #     # choose candidates based on column fitness
                #     parent_a = select_candidate(pop, selection_rate, fitness_function='col')
                #     parent_b = select_candidate(pop, selection_rate, fitness_function='col')
                #     child_a, child_b = self.crossover_columns(parent_a, parent_b)
                # elif f == 1:
                #     # choose candidates based on row fitness
                #     parent_a = select_candidate(pop, selection_rate, fitness_function='row')
                #     parent_b = select_candidate(pop, selection_rate, fitness_function='row')
                #     child_a, child_b = self.crossover_rows(parent_a, parent_b)
                parent_a = select_candidate(pop, selection_rate, fitness_function='overall')
                parent_b = select_candidate(pop, selection_rate, fitness_function='overall')
                child_a, child_b = self.crossover_squares(parent_a, parent_b)

                # Todo Mutate children
                for i in range(0, 5):
                    child_a.config = solver.mutate(child_a, mutation_rate)
                    child_b.config = solver.mutate(child_b, mutation_rate)

                child_a.config = self.fill_immutable_squares(child_a.config)
                child_b.config = self.fill_immutable_squares(child_b.config)

                next_gen.append(child_a)
                next_gen.append(child_b)

            next_gen.sort(key=lambda x: x.determine_fitness(), reverse=True)

            pop = next_gen


def select_candidate(candidates, selection_rate, fitness_function):
    parent_a = candidates[random.randint(0, len(candidates) - 1)]
    parent_b = candidates[random.randint(0, len(candidates) - 1)]

    parent_a_fitness = parent_a.determine_fitness()
    parent_b_fitness = parent_b.determine_fitness()

    if parent_a_fitness > parent_b_fitness:
        fitter = parent_a
        weaker = parent_b
    else:
        fitter = parent_b
        weaker = parent_a

    r = random.uniform(0, 1)
    if r < selection_rate:
        return fitter
    else:
        return weaker


if __name__ == "__main__":
    input = '002000634106000580007300290000000000085001006000750023003000050000000000314002000009080400720040009'

    solver = Solver(input)

    # parent_a = solver.create_candidate()
    # parent_b = solver.create_candidate()
    #
    # solver.crossover_squares(parent_a,parent_b)



    solver.solve(4000, 3000, 0.05, 0.8, 0.4)
